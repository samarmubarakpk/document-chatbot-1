"""
Script to completely clear your Qdrant vector database

⚠️ WARNING: This will DELETE ALL data in your Qdrant collections!

Usage:
    python clear_qdrant_database.py
"""

from qdrant_client import QdrantClient
import sys

# Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

def clear_qdrant_database():
    """Delete all collections from Qdrant"""
    
    print("\n" + "="*80)
    print("  QDRANT DATABASE CLEANUP")
    print("="*80 + "\n")
    
    print("⚠️  WARNING: This will DELETE ALL collections and data!")
    print(f"   Target: {QDRANT_HOST}:{QDRANT_PORT}\n")
    
    # Ask for confirmation
    response = input("Type 'DELETE' to confirm: ")
    
    if response != "DELETE":
        print("\n❌ Cancelled. No data was deleted.")
        return
    
    try:
        print("\n🔧 Connecting to Qdrant...")
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Get all collections
        print("📋 Fetching all collections...")
        collections = client.get_collections()
        
        if not collections.collections:
            print("✅ No collections found. Database is already empty!")
            return
        
        print(f"\nFound {len(collections.collections)} collection(s):")
        for col in collections.collections:
            print(f"   • {col.name}")
        
        # Delete each collection
        print(f"\n🗑️  Deleting collections...")
        for col in collections.collections:
            print(f"   Deleting '{col.name}'...", end=" ")
            client.delete_collection(col.name)
            print("✅")
        
        print(f"\n{'='*80}")
        print("✅ ALL COLLECTIONS DELETED SUCCESSFULLY")
        print("="*80)
        print("\nYour Qdrant database is now completely empty.")
        print("Next step: Upload documents through your application.\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    clear_qdrant_database()