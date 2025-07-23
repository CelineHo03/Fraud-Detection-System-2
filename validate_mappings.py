# Test your fix
import pandas as pd
from app.ml.dataset_adapter import adapt_fraud_dataset

test_data = pd.DataFrame({
    'lat': [40.7128, 34.0522],  # Should NOT map anywhere now
    'merchant': ['Store_A', 'Store_B'],  # Should map with lower confidence
    'unix_time': [1640995200, 1641081600],  # Should map to TransactionDT
    'amount': [25.50, 75.20]
})

result = adapt_fraud_dataset(test_data, verbose=True)

print("🔍 TESTING FIXES:")
for mapping in result.column_mappings:
    if mapping.original_name == 'lat':
        print(f"❌ STILL BROKEN: lat mapped to {mapping.mapped_name}")
    else:
        status = "✅" if mapping.confidence <= 0.85 or mapping.original_name not in ['merchant'] else "⚠️"
        print(f"{status} '{mapping.original_name}' -> {mapping.mapped_name} ({mapping.confidence:.0%})")