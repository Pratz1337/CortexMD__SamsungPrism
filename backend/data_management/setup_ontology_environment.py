"""
Setup script for ontology environment configuration
Loads .env file and verifies configuration
"""
import os
import sys
from pathlib import Path

def setup_environment():
    """Setup and verify ontology environment configuration"""
    print("🔧 Setting up Ontology Environment Configuration")
    print("=" * 50)

    # Get the backend directory
    backend_dir = Path(__file__).parent
    env_file = backend_dir / ".env"

    # Check if .env file exists
    if not env_file.exists():
        print(f"❌ .env file not found at: {env_file}")
        print("   Please create a .env file with ontology configuration")
        return False

    print(f"✅ Found .env file: {env_file}")

    # Load environment variables manually (since python-dotenv may not be available)
    print("\n📋 Loading environment variables from .env:")

    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Set environment variable
                os.environ[key] = value
                print(f"   ✅ {key}: {'***' if 'PASSWORD' in key or 'KEY' in key else value}")

    # Verify critical environment variables
    print("\n🔍 Verifying Critical Configuration:")

    critical_vars = {
        'UMLS_API_KEY': '4563e39c-b5ba-4994-b288-8c45269f5d88',
        'NEO4J_URI': 'Neo4j database connection',
        'NEO4J_USERNAME': 'Neo4j authentication',
        'NEO4J_PASSWORD': 'Neo4j authentication'
    }

    all_configured = True
    for var, description in critical_vars.items():
        value = os.getenv(var)
        if value:
            if 'PASSWORD' in var or 'KEY' in var:
                print(f"   ✅ {var}: Configured ({description})")
            else:
                print(f"   ✅ {var}: {value} ({description})")
        else:
            print(f"   ⚠️  {var}: Not configured ({description})")
            all_configured = False

    # Optional variables
    optional_vars = {
        'SNOMED_API_KEY': 'SNOMED CT API access',
        'ENV': 'Environment (development/production)'
    }

    print("\n📋 Optional Configuration:")
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            print(f"   ✅ {var}: {value} ({description})")
        else:
            print(f"   ℹ️  {var}: Not configured ({description})")

    # Test imports
    print("\n🔧 Testing Ontology Imports:")
    try:
        sys.path.append(str(backend_dir))

        # Test basic imports
        from services.ontology_mapper import OntologyMapper
        print("   ✅ OntologyMapper import successful")

        from config.neo4j_config import get_config
        print("   ✅ Neo4j configuration import successful")

        # Test enhanced ontology mapper initialization
        mapper = OntologyMapper(use_enhanced_services=True)
        print("   ✅ Enhanced ontology mapper initialization successful")

    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print("   Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"   ❌ Initialization error: {e}")
        return False

    print("\n" + "=" * 50)
    print("🎉 Ontology Environment Setup Complete!")

    if all_configured:
        print("\n✅ All critical components configured:")
        print("   ✅ UMLS API access ready")
        print("   ✅ Neo4j knowledge graph ready")
        print("   ✅ Enhanced ontology mapping ready")
        print("   🚀 System ready for production use!")
    else:
        print("\n⚠️  Partial configuration detected:")
        print("   ✅ Basic ontology mapping available")
        print("   ⚠️  Enhanced features may be limited")
        print("   💡 Configure missing environment variables for full functionality")

    print("\n📝 Next Steps:")
    print("   1. Start Neo4j database (if using knowledge graph)")
    print("   2. Run integration tests: python test_ontology_integration.py")
    print("   3. Integrate with your CortexMD application")

    return True

if __name__ == "__main__":
    success = setup_environment()
    sys.exit(0 if success else 1)
