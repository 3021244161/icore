{{/*
Common labels applied to every icore resource.
*/}}
{{- define "icore.labels" -}}
app.kubernetes.io/name: icore
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end -}}

{{/*
Selector labels (must be stable across upgrades).
*/}}
{{- define "icore.selectorLabels" -}}
app.kubernetes.io/name: icore
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Service account name: explicit override, or auto-generated.
*/}}
{{- define "icore.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (printf "icore-%s" .Release.Name) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}
