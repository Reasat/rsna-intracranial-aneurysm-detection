# Add this cell to your notebook to force reload the analysis module
# Copy and paste this code into a new cell in your notebook

# Force reload the analysis module
import importlib
import sys

print("🔄 Force reloading analysis module...")

# Clear the analysis module from cache
if 'analysis' in sys.modules:
    del sys.modules['analysis']
    print("✅ Cleared analysis module from cache")

# Also clear related modules
modules_to_clear = ['inference', 'utils', 'model']
for module_name in modules_to_clear:
    if module_name in sys.modules:
        del sys.modules[module_name]
        print(f"✅ Cleared {module_name} module from cache")

# Re-import the analysis module
try:
    from analysis import AnalysisEngine, VisualizationEngine
    print("✅ Successfully re-imported AnalysisEngine and VisualizationEngine")
except Exception as e:
    print(f"❌ Error re-importing: {e}")

print("🎯 Now you can run your visualization cells!")
