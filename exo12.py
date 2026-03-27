import os
import sys
from email import policy
from email.parser import BytesParser

def extract_image_from_eml(eml_path):
    os.makedirs("attachments", exist_ok=True)

    with open(eml_path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    for part in msg.walk():
        content_type = part.get_content_type()
        filename = part.get_filename()

        print("Type trouvé :", content_type, "| Nom :", filename)

        if content_type.startswith("image/"):
            if not filename:
                ext = content_type.split("/")[-1]
                filename = f"image_jointe.{ext}"

            output_path = os.path.join("attachments", filename)

            payload = part.get_payload(decode=True)
            if payload:
                with open(output_path, "wb") as img_file:
                    img_file.write(payload)

                print(f"Image extraite : {output_path}")
                return output_path

    print("Aucune image trouvée dans le fichier .eml")
    return None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage : python exo12.py exo_12_image.eml")
        sys.exit(1)

    extract_image_from_eml(sys.argv[1])