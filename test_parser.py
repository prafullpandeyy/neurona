from app.parser import extract_text_from_pdf

file_path = "data/uploads/sample.pdf"  # Make sure this file exists

result = extract_text_from_pdf(file_path)

print("✅ File metadata:")
print(result["metadata"])

print("\n🧾 Outline / Table of Contents:")
print(result["outline"][:2])

print("\n📄 First Page Preview:\n")
print(result["pages"][0][:1000])
