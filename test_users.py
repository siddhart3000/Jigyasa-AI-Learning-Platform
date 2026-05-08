from modules.firebase_service import db

users = db.collection("users").stream()

print("\n=== USERS ===\n")

for user in users:
    data = user.to_dict()

    print("ID:", user.id)
    print("Name:", data.get("name"))
    print("Email:", data.get("email"))
    print("Role:", data.get("role"))
    print("-" * 30)