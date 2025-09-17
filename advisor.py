# advisor.py
import sys

# --- try import pandas (required) ---
try:
    import pandas as pd
except Exception as e:
    print("ERROR: pandas not installed. Install with: python -m pip install pandas")
    print("Details:", e)
    sys.exit(1)

# --- Try to import scikit-learn; if not available we'll use fallback logic ---
use_sklearn = True
try:
    from sklearn.preprocessing import LabelEncoder
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, classification_report
except Exception as e:
    print("Note: scikit-learn not available. Falling back to dataset lookup (no ML).")
    use_sklearn = False

# ---------------- Load dataset ----------------
CSV_FILE = "student_courses_large.csv"
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print(f"ERROR: {CSV_FILE} not found. Put the CSV in the same folder as this script.")
    sys.exit(1)

# Basic cleaning: lower-case categorical columns for consistent matching
df['Interest'] = df['Interest'].astype(str).str.strip().str.lower()
df['Goal'] = df['Goal'].astype(str).str.strip().str.lower()
df['Recommended_Course'] = df['Recommended_Course'].astype(str).str.strip()

# ---------------- If sklearn available: train a Decision Tree ----------------
if use_sklearn:
    le_interest = LabelEncoder()
    le_goal = LabelEncoder()
    le_course = LabelEncoder()

    df["Interest_enc"] = le_interest.fit_transform(df["Interest"])
    df["Goal_enc"] = le_goal.fit_transform(df["Goal"])
    df["Course_enc"] = le_course.fit_transform(df["Recommended_Course"])

    X = df[["Interest_enc", "Percentage", "Goal_enc"]]
    y = df["Course_enc"]

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)

    # Optional: show training accuracy (since dataset is synthetic, training accuracy will be high)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print("ML model trained. Training accuracy: {:.2f}%".format(acc * 100))
    # Comment out the next line if output too verbose:
    try:
        print("\nClassification report (sample):")
        print(classification_report(y, y_pred, target_names=le_course.classes_))
    except Exception:
        pass

# ---------------- Fallback predictor (no sklearn) ----------------
def fallback_predict(interest, percentage, goal, top_k=1):
    """
    Strategy:
      1) Try exact (interest & goal) matches and pick closest by percentage (or mode).
      2) If none, try only interest matches.
      3) If still none, return most common course overall.
    Returns top_k recommended courses as list.
    """
    interest = interest.strip().lower()
    goal = goal.strip().lower()

    # exact interest+goal matches
    df_sub = df[(df['Interest'] == interest) & (df['Goal'] == goal)]

    if df_sub.empty:
        df_sub = df[df['Interest'] == interest]

    if df_sub.empty:
        # fallback to most common overall
        top = df['Recommended_Course'].value_counts().index.tolist()[:top_k]
        return top

    # calculate distance by percentage
    df_sub = df_sub.copy()
    df_sub['pct_diff'] = (df_sub['Percentage'] - percentage).abs()
    df_sub = df_sub.sort_values(['pct_diff'])

    # pick top_k distinct course names ordered by how close they are
    seen = []
    for _, row in df_sub.iterrows():
        course = row['Recommended_Course']
        if course not in seen:
            seen.append(course)
        if len(seen) >= top_k:
            break
    return seen

# ---------------- Unified prediction function ----------------
def predict_course(interest, percentage, goal, top_k=1):
    # input sanitization
    interest = str(interest).strip().lower()
    goal = str(goal).strip().lower()
    try:
        percentage = float(percentage)
    except:
        percentage = float(0)

    if use_sklearn:
        # if a label encoder doesn't know this category it will raise; handle gracefully
        try:
            interest_enc = le_interest.transform([interest])[0]
            goal_enc = le_goal.transform([goal])[0]
        except Exception:
            # unknown category -> use fallback
            return fallback_predict(interest, percentage, goal, top_k)

        pred_arr = model.predict([[interest_enc, percentage, goal_enc]])
        try:
            # If top_k > 1, we can try to get nearest neighbors using dataset lookup
            if top_k == 1:
                return [le_course.inverse_transform(pred_arr)[0]]
            else:
                # For top_k results, use fallback to get varied suggestions
                return fallback_predict(interest, percentage, goal, top_k)
        except Exception:
            return fallback_predict(interest, percentage, goal, top_k)
    else:
        return fallback_predict(interest, percentage, goal, top_k)

# ---------------- Simple CLI test (if run from command line) ----------------
if __name__ == "__main__":
    # quick interactive test if user runs script directly from terminal
    print("\n=== Student Course Advisor ===")
    i = input("Interest (ai/web/cyber/iot): ").strip().lower()
    p = input("Percentage: ").strip()
    g = input("Goal (job/research/startup/higher studies): ").strip().lower()

    recs = predict_course(i, p, g, top_k=3)
    print("\nTop recommendations:")
    for idx, r in enumerate(recs, start=1):
        print(f"{idx}. {r}")

    print("\n(You can also integrate this script with a GUI — it will use ML if sklearn is installed.)")
