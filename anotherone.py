f = open("storage.txt", "r")
result = f.readlines()
f.close()

print(result)
ans = ""
found = False

for res in result:

    if "```json" in res:
        found = True
    elif "```" in res:
        found=False
    
    if found and "```json" not in res:
        ans = ans + res

print("###########################")
print(result[0])
print(ans)

import json 
format = json.loads(ans)

with open("storage_maxpoints.txt", "w") as f:
    json.dump(format, f)