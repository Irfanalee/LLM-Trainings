import json

# Your 20 varied examples
raw_data = [
    ("There is a blue manual gate valve on line 100.", '{"tag": "VLV", "type": "gate", "op": "manual", "color": "blue", "line": 100}'),
    ("Line 250 has a red motorized globe valve.", '{"tag": "VLV", "type": "globe", "op": "motorized", "color": "red", "line": 250}'),
    ("Found a green check valve on line 45.", '{"tag": "VLV", "type": "check", "op": "auto", "color": "green", "line": 45}'),
    ("On line 10, there is a yellow ball valve (manual).", '{"tag": "VLV", "type": "ball", "op": "manual", "color": "yellow", "line": 10}'),
    ("Line 500 features a silver butterfly valve with a motor.", '{"tag": "VLV", "type": "butterfly", "op": "motorized", "color": "silver", "line": 500}'),
    ("A black relief valve is located on line 12.", '{"tag": "VLV", "type": "relief", "op": "auto", "color": "black", "line": 12}'),
    ("There's a white needle valve on line 88 (manual).", '{"tag": "VLV", "type": "needle", "op": "manual", "color": "white", "line": 88}'),
    ("Line 1001 contains a motorized orange plug valve.", '{"tag": "VLV", "type": "plug", "op": "motorized", "color": "orange", "line": 1001}'),
    ("Manual bronze gate valve observed on line 7.", '{"tag": "VLV", "type": "gate", "op": "manual", "color": "bronze", "line": 7}'),
    ("Line 33 has a purple automatic diaphragm valve.", '{"tag": "VLV", "type": "diaphragm", "op": "auto", "color": "purple", "line": 33}'),
    ("A grey solenoid valve is on line 99.", '{"tag": "VLV", "type": "solenoid", "op": "auto", "color": "grey", "line": 99}'),
    ("Check the manual pink ball valve on line 14.", '{"tag": "VLV", "type": "ball", "op": "manual", "color": "pink", "line": 14}'),
    ("Line 67: motorized brown globe valve.", '{"tag": "VLV", "type": "globe", "op": "motorized", "color": "brown", "line": 67}'),
    ("Small gold manual valve found on line 2.", '{"tag": "VLV", "type": "unknown", "op": "manual", "color": "gold", "line": 2}'),
    ("Line 400 has a cyan relief valve.", '{"tag": "VLV", "type": "relief", "op": "auto", "color": "cyan", "line": 400}'),
    ("Identify the motorized teal butterfly valve on line 19.", '{"tag": "VLV", "type": "butterfly", "op": "motorized", "color": "teal", "line": 19}'),
    ("Line 5: manual navy gate valve.", '{"tag": "VLV", "type": "gate", "op": "manual", "color": "navy", "line": 5}'),
    ("Automatic lime check valve on line 55.", '{"tag": "VLV", "type": "check", "op": "auto", "color": "lime", "line": 55}'),
    ("Line 800 contains a motorized maroon plug valve.", '{"tag": "VLV", "type": "plug", "op": "motorized", "color": "maroon", "line": 800}'),
    ("Manual beige needle valve spotted on line 22.", '{"tag": "VLV", "type": "needle", "op": "manual", "color": "beige", "line": 22}')
]

# Write to JSONL format
with open("dataset.jsonl", "w") as f:
    for text, json_out in raw_data:
        example = {
            "instruction": "Extract equipment asset details from the sentence into a structured JSON object.",
            "input": text,
            "output": json_out
        }
        f.write(json.dumps(example) + "\n")

print("Successfully created dataset.jsonl with 20 examples.")