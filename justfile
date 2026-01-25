# Client regeneration from llm-engine OpenAPI schema

# Configuration
llm_engine_repo := "scaleapi/llm-engine"
default_branch := "main"
schema_path := "model-engine/specs/openapi-3.0.json"
generator_version := "6.4.0"

# Fetch the OpenAPI schema from llm-engine repo
fetch-schema branch=default_branch:
    @echo "Fetching OpenAPI 3.0 schema from {{llm_engine_repo}} (branch: {{branch}})..."
    curl -sSL "https://raw.githubusercontent.com/{{llm_engine_repo}}/{{branch}}/{{schema_path}}" -o openapi.json
    @echo "Schema saved to openapi.json"

# Generate client code from openapi.json
generate:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -f openapi.json ]; then
        echo "Error: openapi.json not found. Run 'just fetch-schema' first."
        exit 1
    fi
    echo "Generating client with OpenAPI Generator {{generator_version}}..."
    docker run --rm \
        -v "$(pwd):/local" \
        openapitools/openapi-generator-cli:v{{generator_version}} generate \
        -i /local/openapi.json \
        -g python \
        -o /local \
        --package-name launch.api_client \
        --additional-properties=generateSourceCodeOnly=true
    echo "Client generated. Review changes with 'git diff'"

# Fetch schema and regenerate client
regenerate branch=default_branch: (fetch-schema branch) generate

# Show current schema source info
info:
    @echo "Schema source: https://github.com/{{llm_engine_repo}}/blob/{{default_branch}}/{{schema_path}}"
    @echo "Generator version: {{generator_version}}"
    @test -f openapi.json && echo "Local schema: openapi.json (exists)" || echo "Local schema: openapi.json (not found)"
