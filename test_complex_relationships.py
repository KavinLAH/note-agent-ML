import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from ml.extraction import LLMExtractor

def test_complex_relations():
    print("\n" + "="*60)
    print(" TESTING COMPLEX RELATIONSHIP EXTRACTION")
    print("="*60 + "\n")

    # Verify API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("✗ GROQ_API_KEY not set. Skipping test.")
        return

    extractor = LLMExtractor()

    # Craft text with all 6 relation types
    # 1. Causes/Supports: "Increased CO2 causes warming."
    # 2. Contradicts: "Some argue CO2 does not cause warming."
    # 3. Refines: "Specifically, anthropogenic CO2 is the main driver."
    # 4. DependsOn: "Global warming depends on the greenhouse effect."
    # 5. SameAs: "The greenhouse effect is the same as atmospheric heat trapping."
    complex_text = """
    Increased CO2 concentrations in the atmosphere causes global warming.
    This claim is supports by satellite data from NASA.
    However, a small group of critics contradicts this, claiming climate change is natural.
    Specifically, anthropogenic carbon emissions refine our understanding as the primary driver.
    Global warming depends on the greenhouse effect to occur.
    The greenhouse effect is essentially the same as atmospheric heat trapping.
    """

    print(f"--- Input Text ---\n{complex_text.strip()}\n------------------\n")

    result = extractor.extract(complex_text)

    print(f"\nExtracted Objects: {len(result.objects)}")
    for obj in result.objects:
        print(f"  {obj.id}: [{obj.type}] {obj.canonical_text}")

    print(f"\nExtracted Links: {len(result.links)}")
    found_types = set()
    for link in result.links:
        print(f"  {link.source_id} --[{link.type}]--> {link.target_id}")
        found_types.add(link.type)

    # Check for core types
    required_types = {'Supports', 'Contradicts', 'Refines', 'DependsOn', 'SameAs', 'Causes'}
    missing = required_types - found_types
    
    print("\n" + "="*60)
    if not missing:
        print("✓ SUCCESS: All complex relationship types were detected!")
    else:
        print(f"⚠ PARTIAL SUCCESS: Detected types: {found_types}")
        print(f"  Missing types: {missing}")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_complex_relations()
