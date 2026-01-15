"""
Test script for the Agentic RAG API.
"""
import asyncio
import httpx
from pprint import pprint


async def test_health():
    """Test the health endpoint."""
    print("\n=== Testing Health Endpoint ===")
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/health")
        print(f"Status Code: {response.status_code}")
        pprint(response.json())


async def test_query():
    """Test the query endpoint."""
    print("\n=== Testing Query Endpoint ===")
    
    query = "Go to https://crawl4ai.com/ and tell me about its main features and pricing."
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "http://localhost:8000/query",
            json={"query": query}
        )
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nSuccess: {data['success']}")
            print(f"\nNumber of messages: {len(data['messages'])}")
            
            if data.get('final_answer'):
                print("\n=== Final Answer ===")
                print(data['final_answer'])
            else:
                print("\n=== All Messages ===")
                for i, msg in enumerate(data['messages'], 1):
                    print(f"\n--- Message {i} ({msg['role']}) ---")
                    print(msg['content'][:500] + "..." if len(msg['content']) > 500 else msg['content'])
        else:
            print("Error:", response.text)


async def main():
    """Run all tests."""
    try:
        await test_health()
        await test_query()
    except httpx.ConnectError:
        print("\n❌ Error: Could not connect to the API.")
        print("Make sure the server is running with: python -m my_agent.api.main")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
