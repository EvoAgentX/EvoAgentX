from supabase import create_client, Client
import os
from dotenv import load_dotenv
import asyncio
from supabase import create_client, Client

load_dotenv(os.path.join(os.path.dirname(__file__), "server", 'app.env'))
SUPABASE_URL_REQUIREMENT = os.getenv("SUPABASE_URL_REQUIREMENT")
SUPABASE_KEY_REQUIREMENT = os.getenv("SUPABASE_KEY_REQUIREMENT")
project_id = "zw7nnyv"
requirement_id = "t78kwdm"
file_path =  f"projects/{project_id}/requirements/{requirement_id}/requirement.md"


async def main():
    client: Client = create_client(SUPABASE_URL_REQUIREMENT, SUPABASE_KEY_REQUIREMENT)
    
    response = (
        client.storage
        .from_("create-x")
        .download(file_path)
    )
    
    print(response.decode("utf-8"))
    

if __name__ == "__main__":
    asyncio.run(main())