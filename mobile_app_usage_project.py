"""Project 24: Mobile App Usage & User Engagement Analysis

Requirements: pandas, numpy, matplotlib (optional for charts)
Run: python mobile_app_usage_project.py
Place 24_mobile_app_usage.csv in the same folder as this script.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).with_name("24_mobile_app_usage.csv")

# Map common column-name variations to the project's standard names.
ALIASES = {
    "Session_ID": ["session_id", "sessionid", "session id"],
    "User_ID": ["user_id", "userid", "user id"],
    "Session_Date": ["session_date", "date", "session date"],
    "Session_Duration_Mins": ["session_duration_mins", "session_duration_min", "session_duration", "duration_mins", "duration_min"],
    "User_Rating": ["user_rating", "rating", "user rating"],
    "App_Feature": ["app_feature", "feature", "app feature", "feature_used"],
    "Device_Type": ["device_type", "device", "device type"],
    "Age_Group": ["age_group", "age group", "agegroup"],
}


def standardize_columns(df):
    normalized = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    rename = {}
    for standard, aliases in ALIASES.items():
        candidates = [standard.lower(), standard.lower().replace("_", " ")] + aliases
        for candidate in candidates:
            key = candidate.strip().lower().replace(" ", "_")
            if key in normalized:
                rename[normalized[key]] = standard
                break
    return df.rename(columns=rename)



def average_duration(values):
    """Return mean session duration, ignoring missing values."""
    return float(pd.Series(values).dropna().mean())



def feature_duration_stats(df, feature):
    """NumPy: mean, median, and standard deviation for one feature."""
    durations = df.loc[df["App_Feature"] == feature, "Session_Duration_Mins"].dropna().to_numpy(dtype=float)
    if durations.size == 0:
        return {"Feature": feature, "Mean": np.nan, "Median": np.nan, "Std": np.nan, "Count": 0}
    return {
        "Feature": feature,
        "Mean": float(np.mean(durations)),
        "Median": float(np.median(durations)),
        "Std": float(np.std(durations, ddof=1)) if durations.size > 1 else 0.0,
        "Count": int(durations.size),
    }



class AppUser:
    """Simple user record containing a user ID and device category."""
    def __init__(self, user_id, device_type):
        self.user_id = user_id
        self.device_type = device_type




def recommendations():
    return [
        "Personalize the home screen based on the user's most-used features.",
        "Reduce loading time and improve responsiveness, especially on frequently used screens.",
        "Use clear onboarding and short feature tutorials to help new users discover tools.",
        "Collect optional feedback after sessions and investigate low ratings for specific features.",
        "Send relevant, user-controlled reminders at suitable times rather than excessive notifications.",
    ]



def main():
    if not DATA_FILE.exists():
        print(f"Dataset not found: {DATA_FILE.name}")
        print("Place the CSV file in the same folder as this Python script and run again.")
        return


    # 1. Load and inspect data
    df = pd.read_csv(DATA_FILE)
    df = standardize_columns(df)
    print("\n=== 1. DATASET OVERVIEW ===")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head())
    print("\nData types:\n", df.dtypes)
    print("\nDescriptive statistics:\n", df.describe(include="all"))

    # Verify columns required by the assignment
    required = ["Session_ID", "User_ID", "Session_Date", "Session_Duration_Mins", "User_Rating", "App_Feature", "Device_Type", "Age_Group"]
    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        print("\nSome expected columns were not found:", missing_columns)
        print("Please check the CSV headers. The script recognizes common header variations.")
        print("Analysis will continue for the available columns where possible.")

    # 2. Missing values and duplicate session records
    print("\n=== 2. MISSING VALUES & DUPLICATES ===")
    print("Missing values by column:\n", df.isna().sum())
    if "Session_ID" in df.columns:
        print("Duplicate Session_ID records:", int(df["Session_ID"].duplicated().sum()))
    duplicate_rows = int(df.duplicated().sum())
    print("Duplicate entire rows:", duplicate_rows)
    df = df.drop_duplicates().copy()
    if "Session_ID" in df.columns:
        df = df.drop_duplicates(subset="Session_ID", keep="first").copy()

    # 3. Missing duration/rating: duration median (robust), rating median
    print("\n=== 3. CLEANING MISSING VALUES ===")
    if "Session_Duration_Mins" in df.columns:
        df["Session_Duration_Mins"] = pd.to_numeric(df["Session_Duration_Mins"], errors="coerce")
        duration_median = df["Session_Duration_Mins"].median()
        df["Session_Duration_Mins"] = df["Session_Duration_Mins"].fillna(duration_median)
        # Negative duration is invalid; replace with median as well.
        if pd.notna(duration_median):
            df.loc[df["Session_Duration_Mins"] < 0, "Session_Duration_Mins"] = duration_median
        print(f"Session duration missing/invalid values filled using median ({duration_median}).")
    if "User_Rating" in df.columns:
        df["User_Rating"] = pd.to_numeric(df["User_Rating"], errors="coerce")
        rating_median = df["User_Rating"].median()
        df["User_Rating"] = df["User_Rating"].fillna(rating_median)
        print(f"Missing user ratings filled using median ({rating_median}).")

    # 4. Date conversion and calendar features
    print("\n=== 4. DATE FEATURES ===")
    if "Session_Date" in df.columns:
        df["Session_Date"] = pd.to_datetime(df["Session_Date"], errors="coerce")
        df["Month"] = df["Session_Date"].dt.month
        df["Weekday"] = df["Session_Date"].dt.day_name()
        df["Hour"] = df["Session_Date"].dt.hour
        print(df[["Session_Date", "Month", "Weekday", "Hour"]].head())

    # 5. Overall metrics
    print("\n=== 5. OVERALL SESSION METRICS ===")
    total_sessions = len(df)
    unique_users = df["User_ID"].nunique() if "User_ID" in df.columns else np.nan
    avg_duration = df["Session_Duration_Mins"].mean() if "Session_Duration_Mins" in df.columns else np.nan
    print("Total sessions:", total_sessions)
    print("Unique users:", unique_users)
    print("Average session duration (mins):", avg_duration)

    # 6. Top 10 features
    print("\n=== 6. TOP 10 MOST USED FEATURES ===")
    if "App_Feature" in df.columns:
        feature_counts = df["App_Feature"].value_counts()
        top_10_features = feature_counts.head(10)
        most_used_features_list = top_10_features.index.tolist()  # list requirement
        print(top_10_features.to_string())
        print("Stored top features in list:", most_used_features_list)
    else:
        top_10_features = pd.Series(dtype=int)
        most_used_features_list = []

    # 7. Device duration comparison
    print("\n=== 7. AVERAGE SESSION DURATION BY DEVICE ===")
    if "Device_Type" in df.columns and "Session_Duration_Mins" in df.columns:
        device_comparison = df.groupby("Device_Type")["Session_Duration_Mins"].mean().sort_values(ascending=False)
        print(device_comparison.to_string())

    # 8. Age-group session ranking
    print("\n=== 8. SESSIONS BY AGE GROUP ===")
    if "Age_Group" in df.columns:
        age_sessions = df.groupby("Age_Group").size().sort_values(ascending=False)
        print(age_sessions.to_string())

    # 9. Average ratings by feature
    print("\n=== 9. AVERAGE USER RATING BY FEATURE ===")
    if "App_Feature" in df.columns and "User_Rating" in df.columns:
        feature_ratings = df.groupby("App_Feature")["User_Rating"].mean().sort_values(ascending=False)
        print(feature_ratings.to_string())

    # 10. Peak hours and weekdays
    print("\n=== 10. PEAK USAGE TIME ===")
    if "Hour" in df.columns:
        hourly_sessions = df["Hour"].value_counts().sort_index()
        print("Sessions by hour:\n", hourly_sessions.to_string())
        if not hourly_sessions.empty:
            print("Peak hour(s):", hourly_sessions[hourly_sessions == hourly_sessions.max()].index.tolist())
    if "Weekday" in df.columns:
        weekday_sessions = df["Weekday"].value_counts()
        print("Sessions by weekday:\n", weekday_sessions.to_string())
        if not weekday_sessions.empty:
            print("Busiest weekday(s):", weekday_sessions[weekday_sessions == weekday_sessions.max()].index.tolist())

    # 11. Fixed device categories stored in a tuple
    print("\n=== 11. DEVICE CATEGORIES (TUPLE) ===")
    fixed_device_categories = tuple(sorted(df["Device_Type"].dropna().unique())) if "Device_Type" in df.columns else tuple()
    print(fixed_device_categories)

    # 12. Unique users using a set; function to calculate feature average
    print("\n=== 12. UNIQUE USERS & FEATURE AVERAGE FUNCTION ===")
    unique_user_set = set(df["User_ID"].dropna().unique()) if "User_ID" in df.columns else set()
    print("Unique users stored in set:", len(unique_user_set))
    if "App_Feature" in df.columns and "Session_Duration_Mins" in df.columns:
        for feature in most_used_features_list[:10]:
            print(f"Average duration for {feature}: {average_duration(df.loc[df['App_Feature'] == feature, 'Session_Duration_Mins']):.2f} mins")

    # 13. NumPy descriptive statistics
    print("\n=== 13. NUMPY SESSION DURATION STATISTICS ===")
    durations = df["Session_Duration_Mins"].dropna().to_numpy(dtype=float) if "Session_Duration_Mins" in df.columns else np.array([])
    if durations.size:
        print("Mean:", np.mean(durations))
        print("Median:", np.median(durations))
        print("Standard deviation:", np.std(durations))
    else:
        print("No numeric duration values available.")
    if "App_Feature" in df.columns and "Session_Duration_Mins" in df.columns:
        stats = pd.DataFrame([feature_duration_stats(df, f) for f in df["App_Feature"].dropna().unique()])
        print("\nFeature duration statistics:\n", stats.to_string(index=False))

    # 14. IQR outliers and engagement labels
    print("\n=== 14. OUTLIERS & ENGAGEMENT LABELS ===")
    if durations.size:
        q1, q3 = np.percentile(durations, [25, 75])
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = df[(df["Session_Duration_Mins"] < lower_bound) | (df["Session_Duration_Mins"] > upper_bound)]
        print(f"IQR lower bound: {lower_bound:.2f}; upper bound: {upper_bound:.2f}")
        print("Unusually long/short sessions detected:", len(outliers))
        # Documented rule: Low < Q1; Medium from Q1 through Q3; High > Q3.
        q1_all, q3_all = np.percentile(durations, [25, 75])
        df["Engagement_Level"] = np.where(df["Session_Duration_Mins"] < q1_all, "Low",
                                  np.where(df["Session_Duration_Mins"] > q3_all, "High", "Medium"))
        print("Engagement rule: Low (< Q1), Medium (Q1 to Q3 inclusive), High (> Q3)")
        print(df["Engagement_Level"].value_counts().to_string())

    # 15. User class and recommendations
    print("\n=== 15. APP USER CLASS & RECOMMENDATIONS ===")
    if "User_ID" in df.columns and "Device_Type" in df.columns:
        sample_users = df[["User_ID", "Device_Type"]].dropna().drop_duplicates("User_ID").head(5)
        app_users = [AppUser(row.User_ID, row.Device_Type) for row in sample_users.itertuples(index=False)]
        for user in app_users:
            print(f"AppUser(User_ID={user.user_id}, Device_Type={user.device_type})")
    print("Recommendations:")
    for number, recommendation in enumerate(recommendations(), start=1):
        print(f"{number}. {recommendation}")

    # Export cleaned data and useful summaries
    output_dir = Path(__file__).with_name("project_outputs")
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "cleaned_mobile_app_usage.csv", index=False)
    if "App_Feature" in df.columns:
        df["App_Feature"].value_counts().rename_axis("Feature").reset_index(name="Sessions").to_csv(output_dir / "feature_popularity.csv", index=False)
    if "Device_Type" in df.columns and "Session_Duration_Mins" in df.columns:
        df.groupby("Device_Type")["Session_Duration_Mins"].mean().reset_index(name="Average_Duration_Mins").to_csv(output_dir / "device_comparison.csv", index=False)
    print(f"\nCleaned dataset and summary CSV files saved in: {output_dir}")


if __name__ == "__main__":
    main()
