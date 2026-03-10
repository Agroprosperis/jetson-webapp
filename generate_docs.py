import json

def generate_example(schema, definitions):
    """Recursively generates dummy JSON data from a Swagger schema."""
    if not schema:
        return {}
    
    # Handle References
    if '$ref' in schema:
        ref_key = schema['$ref'].split('/')[-1]
        return generate_example(definitions.get(ref_key, {}), definitions)
    
    # Handle Objects
    if schema.get('type') == 'object' and 'properties' in schema:
        obj = {}
        for key, prop in schema['properties'].items():
            obj[key] = generate_example(prop, definitions)
        return obj
    
    # Handle Arrays
    if schema.get('type') == 'array' and 'items' in schema:
        return [generate_example(schema['items'], definitions)]
    
    # Basic Types
    t = schema.get('type')
    if t == 'string':
        return schema.get('enum', ["string"])[0]
    if t == 'integer':
        return 0
    if t == 'number':
        return 0.0
    if t == 'boolean':
        return True
    
    return {}


def example_value_for_param(param):
    if 'enum' in param and param['enum']:
        return param['enum'][0]

    param_type = param.get('type')
    if param_type == 'integer':
        return '0'
    if param_type == 'number':
        return '0.0'
    if param_type == 'boolean':
        return 'true'
    return 'string'


def build_sample_url(path, parameters):
    sample_path = path
    query_parts = []

    for param in parameters:
        location = param.get('in')
        name = param.get('name')
        if not name:
            continue

        sample_value = example_value_for_param(param)
        if location == 'path':
            sample_path = sample_path.replace(f"{{{name}}}", str(sample_value))
        elif location == 'query':
            query_parts.append(f"{name}={sample_value}")

    if query_parts:
        return f"http://localhost{sample_path}?" + "&".join(query_parts)
    return f"http://localhost{sample_path}"


def append_parameters_markdown(md, parameters):
    simple_params = [p for p in parameters if p.get('in') != 'body']
    if not simple_params:
        return

    md.append("### Parameters")
    for param in simple_params:
        name = param.get('name', 'unknown')
        location = param.get('in', 'unknown')
        required = 'required' if param.get('required') else 'optional'
        description = param.get('description', '')
        param_type = param.get('type', 'object')
        enum = param.get('enum')

        line = f"- `{name}` ({location}, {param_type}, {required})"
        if enum:
            line += f": allowed values `{', '.join(map(str, enum))}`"
        if description:
            line += f" - {description}"
        md.append(line)
    md.append("")

def main():
    # 1. Load Swagger File
    try:
        with open('auto_swagger.json', 'r') as f:
            swagger = json.load(f)
    except FileNotFoundError:
        print("❌ Error: auto_swagger.json not found.")
        return

    # 2. Start building Markdown
    md = []
    info = swagger.get('info', {})
    md.append(f"# {info.get('title', 'API Documentation')}\n")
    md.append(f"**Version:** {info.get('version', '0.0.0')}\n")
    md.append(f"**Description:** {info.get('description', '')}\n")
    md.append(f"**Terms of Service:** {info.get('termsOfService', '')}\n")
    md.append("---\n")

    definitions = swagger.get('definitions', {})

    # 3. Iterate over Paths and Methods
    paths = swagger.get('paths', {})
    for path, methods in paths.items():
        for method, details in methods.items():
            summary = details.get('summary', 'No summary')
            method_upper = method.upper()
            
            md.append(f"## {summary}")
            md.append(f"**{method_upper}** `{path}`\n")
            md.append(f"{details.get('description', '')}\n")

            parameters = details.get('parameters', [])
            append_parameters_markdown(md, parameters)

            # --- cURL Sample ---
            md.append("### Request Sample")
            md.append("```shell")
            md.append(f'curl -X {method_upper} "{build_sample_url(path, parameters)}" \\')
            md.append('  -H "accept: application/json" \\')

            # Handle Body Parameter for cURL
            body_param = next((p for p in parameters if p.get('in') == 'body'), None)
            
            if body_param and 'schema' in body_param:
                md.append('  -H "Content-Type: application/json" \\')
                body_example = generate_example(body_param['schema'], definitions)
                md.append(f"  -d '{json.dumps(body_example)}'")
            
            md.append("```\n")

            # --- JSON Response ---
            md.append("### Response")
            responses = details.get('responses', {})
            success_resp = responses.get('200') or responses.get('201')
            
            if success_resp:
                desc = success_resp.get('description', 'Success')
                md.append(f"**200 OK**: {desc}")
                
                if 'schema' in success_resp:
                    response_example = generate_example(success_resp['schema'], definitions)
                    md.append("```json")
                    md.append(json.dumps(response_example, indent=2))
                    md.append("```\n")
            else:
                md.append("No 200/201 response defined.\n")

            md.append("---\n")

    # 4. Write to File
    with open('API_README.md', 'w') as f:
        f.write("\n".join(md))
    
    print("✅ Success! API_README.md generated using Python.")

if __name__ == "__main__":
    main()
