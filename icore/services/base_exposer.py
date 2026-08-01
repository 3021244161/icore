"""
icore.services.base_exposer - Abstract base class for service exposers.

All service exposers implement this interface, enabling a consistent
pattern for converting registered workflows into consumable services
across different protocols (MCP, Tool, Streamlit, SSE).

Each exposer:
    - expose():     Converts a workflow class into a ServiceDef
    - list_services(): Returns all exposed services
    - call_service():  Invokes a service by name
    - start()/stop(): Lifecycle management

Design: Adapter Pattern
    The BaseServiceExposer is the target interface. Each concrete exporter
    adapts a workflow (the adaptee) to a specific protocol. The workflow
    itself is unchanged - it only knows about BaseTask and BaseWorkflow.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from icore.engine.base_workflow import BaseWorkflow

logger = logging.getLogger(__name__)


@dataclass
class ServiceDef:
    """
    Standardised service definition.

    A ServiceDef describes one exposed service. It is protocol-agnostic:
    the same workflow exposes the same ServiceDef across all exposers,
    with only the ``protocol`` field varying.

    Attributes:
        name:          Service name (typically equals workflow_name).
        description:   Human-readable description.
        workflow_name: The registered workflow this service wraps.
        input_schema:  JSON Schema for input parameters.
        output_schema: JSON Schema for output results.
        protocol:      Protocol identifier: "mcp", "tool", "streamlit", "sse".
        metadata:      Extra protocol-specific metadata.
    """

    name: str
    description: str
    workflow_name: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    protocol: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseServiceExposer(abc.ABC):
    """
    Abstract base class for service exposers.

    Subclasses implement protocol-specific logic for exposing workflows
    as services. The interface is intentionally minimal:

        - expose() for registration
        - list_services() for discovery
        - call_service() for invocation
        - start()/stop() for lifecycle

    Class Attributes:
        name:     Human-readable name of this exposer type.
        protocol: Protocol identifier ("mcp", "tool", "streamlit", "sse").
    """

    name: ClassVar[str] = ""
    protocol: ClassVar[str] = ""

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def expose(
        self, workflow_cls: type[BaseWorkflow]
    ) -> ServiceDef:
        """
        Expose a workflow class as a service.

        Implementations register the workflow with the protocol-specific
        server/client and return a ServiceDef describing the exposed
        service.

        Args:
            workflow_cls: The BaseWorkflow subclass to expose.

        Returns:
            A ServiceDef describing the exposed service.
        """
        ...

    @abc.abstractmethod
    async def list_services(self) -> list[ServiceDef]:
        """
        List all currently exposed services.

        Returns:
            A list of ServiceDef for all exposed services under
            this exposer.
        """
        ...

    @abc.abstractmethod
    async def call_service(
        self,
        name: str,
        params: dict[str, Any],
        ctx: Any,  # TaskContext (duck-typed to avoid circular imports)
    ) -> dict[str, Any]:
        """
        Call an exposed service by name.

        Args:
            name:   The service name (typically the workflow_name).
            params: Parameters for the workflow invocation.
            ctx:    A TaskContext with injected dependencies.

        Returns:
            A dict containing the workflow result.

        Raises:
            KeyError: If no service exists with the given name.
        """
        ...

    # ------------------------------------------------------------------
    # Optional lifecycle methods
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Start the exposer (bind ports, register routes, etc.).

        Default is a no-op. Subclasses override to start servers.
        """
        logger.info(
            "Starting %s exposer (protocol=%s)",
            self.name,
            self.protocol,
        )

    async def stop(self) -> None:
        """
        Stop the exposer (close ports, deregister routes, etc.).

        Default is a no-op. Subclasses override to stop servers.
        """
        logger.info(
            "Stopping %s exposer (protocol=%s)",
            self.name,
            self.protocol,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, protocol={self.protocol!r})"
