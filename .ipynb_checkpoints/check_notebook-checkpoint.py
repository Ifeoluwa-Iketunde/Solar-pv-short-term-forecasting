import json

with open('model_development.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

print(f"Total cells: {len(notebook['cells'])}")
print("\nCell types:")
for i, cell in enumerate(notebook['cells']):
    cell_type = cell['cell_type']
    if cell_type == 'code':
        source_preview = ''.join(cell['source'][:1])[:50]
        print(f"Cell {i}: {cell_type} - {source_preview}...")
    else:
        source_preview = ''.join(cell['source'][:1])[:50]
        print(f"Cell {i}: {cell_type} - {source_preview}")
