import pandas as pd
from config import OUTPUT_FILE

def make_predictions(model, X_test, ids):
    preds = model.predict(X_test)
    pred_labels = ["AI" if p > 0.5 else "Human" for p in preds.flatten()]
    pd.DataFrame({"ID": ids, "Label": pred_labels}).to_csv(OUTPUT_FILE, sep='\t', index=False)
    print(f"✅ Ficheiro de submissão gerado: {OUTPUT_FILE}")