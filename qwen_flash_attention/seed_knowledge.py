import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initial knowledge base with coding examples
INITIAL_KNOWLEDGE = [
    {
        "text": """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
""",
        "metadata": {
            "type": "code_example",
            "language": "python",
            "topic": "sorting",
            "agent_role": "assistant"
        }
    },
    {
        "text": """
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None
        
    def insert(self, value):
        if not self.root:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)
            
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert_recursive(node.right, value)
""",
        "metadata": {
            "type": "code_example",
            "language": "python",
            "topic": "data_structures",
            "agent_role": "assistant"
        }
    },
    {
        "text": """
def factorial(n):
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)
""",
        "metadata": {
            "type": "code_example",
            "language": "python",
            "topic": "recursion",
            "agent_role": "assistant"
        }
    }
]

async def seed_database():
    """Seed the MongoDB database with initial knowledge."""
    try:
        # Connect to MongoDB
        client = AsyncIOMotorClient("mongodb://localhost:27017")
        db = client.autogen_vectors
        collection = db.team_knowledge

        # Create text search index
        await collection.create_index([("text", "text")])
        logger.info("Created text search index")

        # Insert documents
        result = await collection.insert_many(INITIAL_KNOWLEDGE)
        logger.info(f"Successfully added {len(result.inserted_ids)} documents")

        # Close connection
        client.close()
        logger.info("Database seeded successfully")

    except Exception as e:
        logger.error(f"Error seeding database: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(seed_database())
