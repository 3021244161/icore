"""
icore.prompts - v0.6 Prompt 模板管理与版本化。

提供：
    - ``PromptVariable``  - 模板变量的元信息（名 / 类型 / 默认值 / 是否必填）。
    - ``PromptTemplate``   - 一个 Prompt 模板（Pydantic 模型），含 ``render()``。
    - ``PromptManager``   - YAML 文件加载、版本管理、A/B 测试、热加载。

模块依赖：
    仅依赖标准库 + ``pydantic`` + ``PyYAML``（已在 requirements.txt）。
    ``yaml`` 在方法内懒导入，与项目 ``config.py`` / ``bootstrap.py`` 一致。

设计原则：
    - **不引入** Jinja2，模板渲染用自实现的 ``{var}`` 替换
    - 所有日志走标准库 ``logging``（**不**使用 loguru）
    - 模板版本存储为 ``config/prompts/<name>/v<version>.yaml``

文件布局示例::

    config/prompts/
    ├── summarizer/
    │   ├── v1.yaml
    │   └── v2.yaml
    └── rag_answer/
        └── v1.yaml
"""

from __future__ import annotations

import logging
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from icore.exceptions import ValidationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PromptVariable / PromptTemplate
# ---------------------------------------------------------------------------


# 支持的变量类型 → Python 类型映射
_VAR_TYPE_MAP: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
}

# 匹配模板中的 {var} 占位符；不匹配 {{ }}（用于字面量花括号转义）
# var 名仅允许字母、数字、下划线
_PLACEHOLDER_PATTERN = re.compile(r"\{(\w+)\}")


class PromptVariable(BaseModel):
    """
    Prompt 模板中的一个变量定义。

    Attributes:
        name:        变量名（与模板中 ``{name}`` 占位符一致）。
        type:        变量类型，取值 ``"str" | "int" | "float" | "bool" | "list"``。
        default:     默认值。未提供时为 ``None``。
        required:    是否必填。``True`` 时调用方未传且无 default 则渲染报错。
        description: 人类可读描述。
    """

    name: str
    type: str = "str"
    default: Any = None
    required: bool = True
    description: str = ""

    def validate_value(self, value: Any) -> Any:
        """
        校验并转换为声明的类型。

        Args:
            value: 调用方传入的原始值。

        Returns:
            转换后的值。

        Raises:
            ValidationError: 类型不符且无法转换。
        """
        expected_type = _VAR_TYPE_MAP.get(self.type)
        if expected_type is None:
            raise ValidationError(
                f"Unknown variable type '{self.type}' for variable '{self.name}'"
            )

        if value is None:
            return None

        # bool 必须在 int 之前判断（Python 中 bool 是 int 子类）
        if expected_type is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                low = value.strip().lower()
                if low in ("true", "1", "yes", "y"):
                    return True
                if low in ("false", "0", "no", "n", ""):
                    return False
                raise ValidationError(
                    f"Variable '{self.name}': cannot convert {value!r} to bool"
                )
            if isinstance(value, int):
                return bool(value)
            raise ValidationError(
                f"Variable '{self.name}': cannot convert {type(value).__name__} to bool"
            )

        if isinstance(value, expected_type):
            return value

        # 尝试字符串 → 数字转换
        if expected_type in (int, float) and isinstance(value, str):
            try:
                return expected_type(value)
            except ValueError as e:
                raise ValidationError(
                    f"Variable '{self.name}': cannot convert {value!r} to {self.type}"
                ) from e

        if expected_type is list and isinstance(value, (tuple, set)):
            return list(value)

        if expected_type is str:
            # 容错：把任何标量转成字符串
            return str(value)

        raise ValidationError(
            f"Variable '{self.name}': expected {self.type}, got {type(value).__name__}"
        )


class PromptTemplate(BaseModel):
    """
    Prompt 模板。

    模板文本用 ``{variable}`` 表示占位符（与 Python ``str.format`` 类似
    但不引入完整 format spec），``{{`` / ``}}`` 表示字面量花括号。

    Attributes:
        name:       模板名（与目录名一致）。
        version:    版本号（>=1）。
        template:   模板字符串。
        variables:  变量定义列表。
        description: 模板描述。
        tags:       标签列表。
        created_at: 创建时间（UTC，自动填充）。
    """

    name: str
    version: int = Field(..., ge=1)
    template: str
    variables: list[PromptVariable] = Field(default_factory=list)
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------

    def render(self, **kwargs: Any) -> str:
        """
        渲染模板，注入变量。

        渲染规则：
            - 对每个 ``PromptVariable``：
                * 若 ``kwargs`` 中提供了值，校验并转换类型
                * 否则使用 ``default``（若有）
                * 否则若 ``required=True`` 抛 ``ValidationError``
                * 否则（optional 无默认）跳过占位符替换（保留 ``{name}`` 原文）
            - ``{{`` / ``}}`` 转义为字面量 ``{`` / ``}``
            - 不在 ``variables`` 中声明的 ``{xxx}`` 占位符保持原样（便于
              留给下游再做一次渲染，也避免误伤模板中的字面量文本）

        Args:
            **kwargs: 变量值。

        Returns:
            渲染后的字符串。

        Raises:
            ValidationError: required 变量缺失或类型校验失败。
        """
        # 1. 解析变量值
        values: dict[str, Any] = {}
        declared_names = {v.name for v in self.variables}
        for var in self.variables:
            if var.name in kwargs:
                values[var.name] = var.validate_value(kwargs[var.name])
            elif var.default is not None:
                values[var.name] = var.validate_value(var.default)
            elif var.required:
                raise ValidationError(
                    f"Missing required variable '{var.name}' for template "
                    f"'{self.name}' v{self.version}"
                )
            # optional 无 default → 不放入 values，占位符保留

        # 2. 先做 {{ / }} → 暂时替换为不可冲突的占位符，避免 {{var}} 被二次解析
        LBRACE = "\x00LBRACE\x00"
        RBRACE = "\x00RBRACE\x00"
        text = self.template.replace("{{", LBRACE).replace("}}", RBRACE)

        # 3. 替换已声明变量
        def _replace(match: re.Match) -> str:
            var_name = match.group(1)
            if var_name in declared_names:
                if var_name in values:
                    return str(values[var_name])
                # 声明但无值（optional 无 default）：保留占位符原文
                return match.group(0)
            # 未声明占位符：保留原文
            return match.group(0)

        text = _PLACEHOLDER_PATTERN.sub(_replace, text)

        # 4. 还原字面量花括号
        text = text.replace(LBRACE, "{").replace(RBRACE, "}")
        return text


# ---------------------------------------------------------------------------
# PromptManager
# ---------------------------------------------------------------------------


class PromptManager:
    """
    Prompt 模板管理器。

    负责从 YAML 文件加载 / 保存模板，并提供版本管理、A/B 测试、热加载。

    YAML 文件路径约定：``<config_dir>/<name>/v<version>.yaml``

    YAML 内容（``name`` 字段不在 YAML 内，由目录名推导）::

        version: 1
        template: |
          Hello {name}!
        variables:
          - name: name
            type: str
            required: true
        description: ...
        tags: [...]

    Attributes:
        config_dir: 模板根目录。
    """

    def __init__(self, config_dir: Union[str, Path] = "config/prompts") -> None:
        self._config_dir: Path = Path(config_dir)
        # name -> {version -> PromptTemplate}（缓存）
        self._cache: dict[str, dict[int, PromptTemplate]] = {}
        self._loaded: bool = False

    @property
    def config_dir(self) -> Path:
        """模板根目录。"""
        return self._config_dir

    # ------------------------------------------------------------------
    # YAML I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _template_to_yaml_dict(template: PromptTemplate) -> dict[str, Any]:
        """把 ``PromptTemplate`` 序列化为 YAML 友好的 dict。"""
        return {
            "version": template.version,
            "template": template.template,
            "variables": [
                {
                    "name": v.name,
                    "type": v.type,
                    "default": v.default,
                    "required": v.required,
                    "description": v.description,
                }
                for v in template.variables
            ],
            "description": template.description,
            "tags": list(template.tags),
        }

    @staticmethod
    def _dict_to_template(name: str, data: dict[str, Any]) -> PromptTemplate:
        """从 YAML 解析出的 dict 构造 ``PromptTemplate``。"""
        if "version" not in data:
            raise ValidationError(
                f"Prompt YAML for '{name}' missing 'version' field"
            )
        if "template" not in data:
            raise ValidationError(
                f"Prompt YAML for '{name}' missing 'template' field"
            )
        version = int(data["version"])
        if version < 1:
            raise ValidationError(
                f"Prompt '{name}' version must be >=1, got {version}"
            )

        variables_data = data.get("variables", []) or []
        variables: list[PromptVariable] = []
        for v in variables_data:
            if not isinstance(v, dict) or "name" not in v:
                raise ValidationError(
                    f"Prompt '{name}' v{version}: variable missing 'name'"
                )
            variables.append(
                PromptVariable(
                    name=v["name"],
                    type=v.get("type", "str"),
                    default=v.get("default"),
                    required=v.get("required", True),
                    description=v.get("description", ""),
                )
            )

        return PromptTemplate(
            name=name,
            version=version,
            template=data["template"],
            variables=variables,
            description=data.get("description", ""),
            tags=list(data.get("tags", []) or []),
        )

    def _path_for(self, name: str, version: int) -> Path:
        """返回 ``<config_dir>/<name>/v<version>.yaml`` 路径。"""
        return self._config_dir / name / f"v{version}.yaml"

    # ------------------------------------------------------------------
    # J2 (Jinja2) I/O — 纯文本模板
    # ------------------------------------------------------------------
    #
    # 约定：``<config_dir>/<name>/v<version>.j2``，文件内容就是
    # Jinja2 模板文本（无 YAML 元数据）。变量用 ``{{ var }}`` 语法。
    # 与 YAML 模板并存：同名的 .yaml 与 .j2 视为两个独立模板，
    # 通过 ``load_from_j2`` / ``render_j2`` 访问。

    def _j2_path_for(self, name: str, version: int) -> Path:
        """返回 ``<config_dir>/<name>/v<version>.j2`` 路径。"""
        return self._config_dir / name / f"v{version}.j2"

    def list_j2_versions(self, name: str) -> list[int]:
        """列出某 J2 模板的所有可用版本号（按升序）。"""
        name_dir = self._config_dir / name
        if not name_dir.exists():
            return []
        versions: list[int] = []
        for j2_file in name_dir.glob("v*.j2"):
            stem = j2_file.stem  # "v1"
            if re.match(r"^v\d+$", stem):
                versions.append(int(stem[1:]))
        return sorted(versions)

    def get_latest_j2_version(self, name: str) -> Optional[int]:
        """获取某 J2 模板的最新版本号；不存在返回 ``None``。"""
        versions = self.list_j2_versions(name)
        return versions[-1] if versions else None

    def load_from_j2(
        self, name: str, version: Optional[int] = None
    ) -> str:
        """
        读取 J2 (Jinja2) 模板的原始文本。

        Args:
            name:    模板名（对应 ``config_dir`` 下的子目录名）。
            version: 版本号；``None`` 表示最新版本。

        Returns:
            模板原始文本（未渲染）。

        Raises:
            ValidationError: 模板或版本不存在。
        """
        if version is None:
            actual = self.get_latest_j2_version(name)
            if actual is None:
                raise ValidationError(
                    f"No J2 prompt template found for name='{name}' "
                    f"in {self._config_dir}"
                )
            version = actual

        path = self._j2_path_for(name, version)
        if not path.exists():
            raise ValidationError(
                f"J2 prompt template not found: {path}"
            )
        with path.open("r", encoding="utf-8") as f:
            return f.read()

    def render_j2(
        self,
        name: str,
        version: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        加载并渲染 J2 (Jinja2) 模板。

        使用 ``jinja2.Environment`` 渲染（lazy import jinja2）。
        变量通过 ``{{ var }}`` 语法注入；未提供的变量在
        ``undefined=jinja2.ChainableUndefined`` 下渲染为空字符串
        （不抛错，便于调试）。

        Args:
            name:    模板名。
            version: 版本号；``None`` 表示最新。
            **kwargs: Jinja2 渲染变量。

        Returns:
            渲染后的字符串。

        Raises:
            ValidationError: 模板不存在。
        """
        raw = self.load_from_j2(name, version)
        try:
            import jinja2  # type: ignore
        except ImportError as e:
            raise ImportError(
                "jinja2 is required to render .j2 prompt templates. "
                "Install with: pip install jinja2"
            ) from e

        env = jinja2.Environment(
            undefined=jinja2.ChainableUndefined,
            autoescape=False,
        )
        try:
            return env.from_string(raw).render(**kwargs)
        except jinja2.TemplateSyntaxError as e:
            raise ValidationError(
                f"J2 template '{name}' v{version} syntax error: {e}"
            ) from e
        except Exception as e:
            raise ValidationError(
                f"J2 template '{name}' v{version} render failed: {e}"
            ) from e

    def scan_j2_templates(
        self,
    ) -> dict[str, list[int]]:
        """
        扫描 ``config_dir``，列出全部 J2 模板及其版本号。

        Returns:
            ``{name: [version, ...]}``，按版本升序。
        """
        result: dict[str, list[int]] = {}
        if not self._config_dir.exists():
            return result
        for name_dir in sorted(self._config_dir.iterdir()):
            if not name_dir.is_dir():
                continue
            name = name_dir.name
            versions = self.list_j2_versions(name)
            if versions:
                result[name] = versions
        return result

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def load_from_yaml(
        self, name: str, version: Optional[int] = None
    ) -> PromptTemplate:
        """
        从 YAML 文件加载指定模板。

        Args:
            name:    模板名（对应 ``config_dir`` 下的子目录名）。
            version: 版本号；``None`` 表示加载最新版本。

        Returns:
            ``PromptTemplate``。

        Raises:
            ValidationError: 模板或版本不存在。
        """
        # 命中缓存（仅 version 指定时）
        if version is not None and name in self._cache and version in self._cache[name]:
            return self._cache[name][version]

        if version is None:
            # 加载最新版本
            actual_version = self.get_latest_version(name)
            if actual_version is None:
                raise ValidationError(
                    f"No prompt template found for name='{name}' in {self._config_dir}"
                )
            version = actual_version

        path = self._path_for(name, version)
        if not path.exists():
            raise ValidationError(
                f"Prompt template not found: {path}"
            )

        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to load prompt templates. "
                "Install with: pip install PyYAML"
            ) from e

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValidationError(
                f"Prompt YAML {path} is not a mapping (got {type(data).__name__})"
            )

        template = self._dict_to_template(name, data)
        self._cache.setdefault(name, {})[version] = template
        return template

    def load_all(self) -> dict[str, list[PromptTemplate]]:
        """
        扫描 ``config_dir``，加载全部模板。

        Returns:
            ``{name: [PromptTemplate, ...]}``，按版本升序。

        若 ``config_dir`` 不存在返回空 dict（不抛错，便于在测试
        环境中初始化空 manager）。
        """
        result: dict[str, list[PromptTemplate]] = {}
        if not self._config_dir.exists():
            self._loaded = True
            return result

        # 清空缓存，确保 reload 时拿到最新数据
        self._cache.clear()
        for name_dir in sorted(self._config_dir.iterdir()):
            if not name_dir.is_dir():
                continue
            name = name_dir.name
            for yaml_file in sorted(name_dir.glob("v*.yaml")):
                # 文件名形如 v1.yaml / v2.yaml
                stem = yaml_file.stem  # "v1"
                if not re.match(r"^v\d+$", stem):
                    continue
                version = int(stem[1:])
                if name in self._cache and version in self._cache[name]:
                    # load_from_yaml 已缓存
                    result.setdefault(name, []).append(self._cache[name][version])
                else:
                    try:
                        tpl = self.load_from_yaml(name, version)
                        result.setdefault(name, []).append(tpl)
                    except Exception as e:
                        logger.warning("Failed to load prompt %s v%d: %s", name, version, e)
            if name in result:
                result[name].sort(key=lambda t: t.version)

        self._loaded = True
        return result

    def save(self, template: PromptTemplate) -> Path:
        """
        保存模板到 YAML 文件。

        会自动创建 ``config_dir/<name>/`` 目录（若不存在）。

        Args:
            template: 待保存的模板。

        Returns:
            保存到的文件路径。

        Raises:
            ValidationError: 模板版本号 < 1。
        """
        if template.version < 1:
            raise ValidationError(
                f"Prompt version must be >=1, got {template.version}"
            )

        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to save prompt templates. "
                "Install with: pip install PyYAML"
            ) from e

        path = self._path_for(template.name, template.version)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self._template_to_yaml_dict(template)
        with path.open("w", encoding="utf-8") as f:
            # allow_unicode=True 避免中文被转义；default_flow_style=False 用 block 风格
            yaml.safe_dump(
                data,
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )

        # 更新缓存
        self._cache.setdefault(template.name, {})[template.version] = template
        logger.info(
            "Saved prompt template '%s' v%d to %s", template.name, template.version, path
        )
        return path

    # ------------------------------------------------------------------
    # 版本查询
    # ------------------------------------------------------------------

    def list_versions(self, name: str) -> list[int]:
        """
        列出某模板的所有可用版本号（按升序）。

        Args:
            name: 模板名。

        Returns:
            版本号列表（升序）。模板不存在则返回空列表。
        """
        # 优先用缓存
        if name in self._cache and self._cache[name]:
            return sorted(self._cache[name].keys())

        name_dir = self._config_dir / name
        if not name_dir.exists():
            return []
        versions: list[int] = []
        for yaml_file in name_dir.glob("v*.yaml"):
            stem = yaml_file.stem
            if re.match(r"^v\d+$", stem):
                versions.append(int(stem[1:]))
        return sorted(versions)

    def get_latest_version(self, name: str) -> Optional[int]:
        """
        获取某模板的最新版本号。

        Args:
            name: 模板名。

        Returns:
            最新版本号；模板不存在返回 ``None``。
        """
        versions = self.list_versions(name)
        return versions[-1] if versions else None

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------

    def render(
        self,
        name: str,
        version: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        加载并渲染模板。

        Args:
            name:    模板名。
            version: 版本号；``None`` 表示最新。
            **kwargs: 渲染变量。

        Returns:
            渲染后的字符串。

        Raises:
            ValidationError: 模板不存在 / required 变量缺失 / 类型校验失败。
        """
        template = self.load_from_yaml(name, version)
        return template.render(**kwargs)

    # ------------------------------------------------------------------
    # A/B 测试
    # ------------------------------------------------------------------

    async def ab_test(
        self,
        name: str,
        v1: int,
        v2: int,
        split: float = 0.5,
    ) -> tuple[PromptTemplate, str]:
        """
        按 ``split`` 比例随机返回 ``v1`` 或 ``v2`` 模板，用于 A/B 测试。

        Args:
            name:  模板名。
            v1:    变体 1 版本号。
            v2:    变体 2 版本号。
            split: 选择 v1 的概率，取值 ``[0, 1]``。``0.5`` 表示 50/50。

        Returns:
            ``(template, variant_label)``，``variant_label`` 为 ``"v1"`` 或 ``"v2"``。

        Raises:
            ValueError: ``split`` 不在 ``[0, 1]`` 或 v1 == v2。
            ValidationError: 任一版本模板不存在。
        """
        if not (0.0 <= split <= 1.0):
            raise ValueError(f"split must be in [0, 1], got {split}")
        if v1 == v2:
            raise ValueError("v1 and v2 must be different versions")

        # 加载两个版本（确保都存在）
        tpl1 = self.load_from_yaml(name, v1)
        tpl2 = self.load_from_yaml(name, v2)

        if random.random() < split:
            chosen, label = tpl1, "v1"
        else:
            chosen, label = tpl2, "v2"

        # 记录 Prometheus 指标：用 try/except 包裹，避免指标写入失败
        # 影响核心 A/B 测试逻辑。
        self._record_ab_test_metric(name, label)

        logger.info(
            "A/B test for prompt '%s': selected %s (v1=%d, v2=%d, split=%.2f)",
            name,
            label,
            v1,
            v2,
            split,
        )
        return chosen, label

    @staticmethod
    def _record_ab_test_metric(prompt_name: str, variant: str) -> None:
        """
        递增 ``icore_prompt_ab_test_total`` counter，按 ``(prompt_name, variant)`` 标签。

        用 try/except 包裹整个调用链，确保可观测性子系统的任何异常
        （observability 模块未初始化、指标注册失败等）都不会影响
        ``ab_test()`` 的核心逻辑。失败时仅记录 debug 日志。
        """
        try:
            from icore.observability import get_metrics_registry

            registry = get_metrics_registry()
            counter = registry.create_counter(
                "icore_prompt_ab_test_total",
                "A/B test variant selection count",
                ("prompt_name", "variant"),
            )
            counter.inc(prompt_name=prompt_name, variant=variant)
        except Exception:
            logger.debug(
                "Failed to record icore_prompt_ab_test_total metric "
                "(prompt_name='%s', variant='%s')",
                prompt_name,
                variant,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # 热加载
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """
        清空缓存并重新扫描 ``config_dir``。

        适用于配置热加载场景：编辑 YAML 后调用此方法生效。
        """
        self._cache.clear()
        self._loaded = False
        self.load_all()
        logger.info(
            "PromptManager reloaded from %s", self._config_dir
        )


__all__ = [
    "PromptVariable",
    "PromptTemplate",
    "PromptManager",
]