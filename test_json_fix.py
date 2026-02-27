import json
import re

def fix_json(text):
    # try to parse
    try:
        return json.loads(text)
    except:
        pass
    
    # look for array or obj
    start = text.find("{")
    if start == -1:
        start = text.find("[")
    if start == -1: return None
    
    json_str = text[start:]
    
    # close unclosed strings
    # a naive way: if there's an unescaped odd number of quotes
    # actually, just removing the last incomplete key/value pair might be easier
    # Let's try appending a quote if odd number
    quote_count = len(re.findall(r'(?<!\\)"', json_str))
    if quote_count % 2 != 0:
        json_str += '"'
        
    # fix trailing commas
    json_str = re.sub(r',\s*$', '', json_str)
        
    # close arrays and objects
    open_braces = json_str.count("{") - json_str.count("}")
    open_brackets = json_str.count("[") - json_str.count("]")
    
    if open_braces > 0:
        json_str += "}" * open_braces
    if open_brackets > 0:
        json_str += "]" * open_brackets
        
    try:
        return json.loads(json_str)
    except Exception as e:
        print("Still failed:", e)
        return None

bad_json = """[
  {
    "text": "Steve Jobs",
    "start": 0,
    "end": 10,
    "type_guess": "Person"
  },
  {
    "text": "Apple",
    "start": 11,
    "end": 16,
    "type_guess": "Organization"
  },
  {
    "text": "iPad",
    "star"""

print(fix_json(bad_json))
