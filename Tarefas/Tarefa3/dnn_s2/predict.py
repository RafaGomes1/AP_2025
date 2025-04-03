import pandas as pd
import numpy as np
from config import OUTPUT_FILE

def make_predictions(model, X_test, ids):
    preds = model.predict(X_test)
    pred_labels = ["AI" if p > 0.5 else "Human" for p in preds.flatten()]
    output_df = pd.DataFrame({
        "ID": ids,
        "Label": pred_labels
    })
    output_df.to_csv(OUTPUT_FILE, sep='\t', index=False)
    print(f"✅ Ficheiro de submissão gerado: {OUTPUT_FILE}")
