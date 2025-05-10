import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import warnings

#Run streamlit app through terminal via 'streamlit run plant_ui_recommender.py'

st.set_page_config(layout="wide", page_title="Plant Recommender", page_icon="🌱")
# clear/clean up terminal
warnings.filterwarnings('ignore', category=UserWarning, message='X does not have valid feature names') 
warnings.filterwarnings('ignore', category=FutureWarning)

NUM_RECOMMENDATIONS_TO_SHOW = 5
METRIC_FOR_KNN = 'cosine'
NO_PREFERENCE_STR = "No Preference / Any"

@st.cache_data #cache since loading large database, do not want to recall everytime we run the app
def load_and_preprocess_data():
    try:
        df = pd.read_csv('usda_cleaned_plant_data.csv')
    except FileNotFoundError:
        st.error("Error: usda_cleaned_plant_data.csv not found.")
        return None, None, None, None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None, None, None, None

    df_processed = df.copy()
    df_processed['pH_Minimum'].fillna(7.0, inplace=True)
    df_processed['pH_Maximum'].fillna(7.0, inplace=True)
    df_processed['Temperature_Minimum_F'].fillna(0, inplace=True)
    ordinal_features_impute = ['Growth_Rate', 'Lifespan', 'Toxicity', 'Drought_Tolerance',
                               'Hedge_Tolerance', 'Moisture_Use', 'Salinity_Tolerance',
                               'Shade_Tolerance']
    for col in ordinal_features_impute:
        if col in df_processed.columns and df_processed[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df_processed[col]):
                median_val = df_processed[col].median()
                df_processed[col].fillna(median_val, inplace=True)

    all_cols_except_id_name = [col for col in df_processed.columns if col not in ['id', 'Scientific_Name_x']]
    for col in all_cols_except_id_name:
        if df_processed[col].isnull().any():
            if col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col]):
                if df_processed[col].nunique() < 10 or df_processed[col].dtype == 'float64':
                     df_processed[col].fillna(df_processed[col].median(), inplace=True)
                else:
                     df_processed[col].fillna(0, inplace=True)
            else:
                df_processed[col].fillna(0, inplace=True)

    identifiers = ['id', 'Scientific_Name_x']
    feature_cols = [col for col in df_processed.columns if col not in identifiers]
    features_df = df_processed[feature_cols].copy()
    for col in features_df.columns:
        if not pd.api.types.is_numeric_dtype(features_df[col]):
            try: features_df[col] = pd.to_numeric(features_df[col])
            except ValueError: features_df[col] = 0

    default_values = {}
    for col in features_df.columns:
        # For one-hot, default is 0. For others, mean/median.
        is_one_hot_like = features_df[col].isin([0, 1]).all() and \
                          col not in ordinal_features_impute and \
                          col not in ['pH_Minimum', 'pH_Maximum', 'Temperature_Minimum_F']

        if is_one_hot_like:
            default_values[col] = 0.0
        elif features_df[col].nunique() < 10 and features_df[col].min() >=0 :
            default_values[col] = features_df[col].median()
        else:
            default_values[col] = features_df[col].mean()

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df.values)

    one_hot_groups = {
        'Category': [c for c in df.columns if c.startswith('Category_')],
        'Family': [c for c in df.columns if c.startswith('Family_')],
        'Growth_Habit': [c for c in df.columns if c.startswith('Growth_Habit_')],
        'Native_Status': [c for c in df.columns if c.startswith('Native_Status_')],
        'Active_Growth_Period': [c for c in df.columns if c.startswith('Active_Growth_Period_')],
        'Fall_Conspicuous': [c for c in df.columns if c.startswith('Fall_Conspicuous_')],
        'Flower_Color': [c for c in df.columns if c.startswith('Flower_Color_')],
        'Flower_Conspicuous': [c for c in df.columns if c.startswith('Flower_Conspicuous_')],
        'Fruit_Conspicuous': [c for c in df.columns if c.startswith('Fruit_Conspicuous_')],
        'Bloom_Period': [c for c in df.columns if c.startswith('Bloom_Period_')],
        'Fire_Resistance': [c for c in df.columns if c.startswith('Fire_Resistance_')]
    }
    return df, features_df, features_scaled, scaler, default_values, one_hot_groups, feature_cols, ordinal_features_impute

df_original, features_df, features_scaled, scaler, default_values, one_hot_groups, feature_cols, ordinal_features_impute = load_and_preprocess_data()


# KNN Recommendation Function (manual distance calculation since scikit library wasn't working)
def recommend_plants_knn_ideal_profile(criteria, k=10, metric='cosine'):
    if df_original is None: return pd.DataFrame()
    query_vector = default_values.copy()
    # st.write("Constructing ideal plant profile based on your criteria:") # Debug

    for key, value in criteria.items():
        if value is None: # skip if user selected "No Preference" for this specific criteria
            continue
        if key in query_vector and key not in one_hot_groups:
            if key == 'pH_Minimum':
                if 'pH_Maximum' in criteria and criteria['pH_Maximum'] is not None:
                    try:
                        min_val = float(value)
                        max_val = float(criteria['pH_Maximum'])
                        mid_ph = (min_val + max_val) / 2
                        query_vector['pH_Minimum'] = mid_ph
                        query_vector['pH_Maximum'] = mid_ph
                    except ValueError: pass
                else:
                    try: query_vector[key] = float(value)
                    except ValueError: pass
            elif key == 'pH_Maximum':
                if 'pH_Minimum' not in criteria or criteria.get('pH_Minimum') is None:
                    try: query_vector[key] = float(value)
                    except ValueError: pass
            else:
                try: query_vector[key] = float(value)
                except (ValueError, TypeError): pass
        elif key in one_hot_groups:
            desired_one_hot_cols_for_group = value
            for col_in_group in one_hot_groups[key]:
                if col_in_group in query_vector: query_vector[col_in_group] = 0.0
            if desired_one_hot_cols_for_group:
                for desired_col in desired_one_hot_cols_for_group:
                    if desired_col in query_vector:
                        query_vector[desired_col] = 1.0
        # else: st.warning(f"Criterion key '{key}' not found. Ignoring for query.") # Debug

    query_df = pd.DataFrame([query_vector], columns=feature_cols)
    for col in feature_cols:
        if col not in query_df.columns: query_df[col] = default_values.get(col, 0)
    query_df = query_df[feature_cols]
    for col in query_df.columns:
        if query_df[col].isnull().any(): query_df[col].fillna(default_values.get(col, 0), inplace=True)
    query_unscaled = query_df.values

    try:
        query_scaled = scaler.transform(query_unscaled)
        if np.isnan(query_scaled).any() or np.isinf(query_scaled).any():
            st.error("Error: Scaled query vector NaN/Inf.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error scaling query: {e}"); return pd.DataFrame()

    # st.write(f"\nCalculating {metric} distances manually...") # Debug
    try:
        if query_scaled.shape[1] != features_scaled.shape[1]:
            st.error(f"Shape mismatch! QF: {query_scaled.shape[1]}, DF: {features_scaled.shape[1]}")
            return pd.DataFrame()
        distances = pairwise_distances(query_scaled, features_scaled, metric=metric)
        distances_flat = distances.flatten()
        nearest_indices = np.argsort(distances_flat)[:k]
        # st.write(f"Found {len(nearest_indices)} closest candidates.") # Debug
    except Exception as e:
        st.error(f"Error in distance calc/sort: {e}")
        return pd.DataFrame() # Debug

    return df_original.iloc[nearest_indices]

# UI Helper Functions
def get_float_input_st(prompt_label, default_value=None, allow_blank=True):
    # number_input with value=None means it's blank until user enters
    val = st.sidebar.number_input(
        prompt_label,
        value=default_value if default_value is not None else np.nan, # Use np.nan for empty number_input
        step=1.0,
        format="%.1f" if "pH" in prompt_label else "%.0f", # format doesn't apply to None/nan
        key=prompt_label.replace(" ", "_") # Unique key for widget
    )
    return None if np.isnan(val) else val


def get_ordinal_selectbox_st(prompt_label, options_map_num_to_display, default_display_option=NO_PREFERENCE_STR):
    """
    options_map_num_to_display: maps numerical value to display string {0: "Low", 1: "Medium"}
    Returns the numerical value corresponding to the user's selection, or None if "No Preference".
    """
    # Create a list of display options for the selectbox, including "No Preference"
    display_options = [NO_PREFERENCE_STR] + list(options_map_num_to_display.values())

    selected_display_str = st.sidebar.selectbox(prompt_label, options=display_options, index=0) # Default to "No Pref"

    if selected_display_str == NO_PREFERENCE_STR:
        return None # User selected no preference

    # Map display string back to numerical value
    for num_val, display_str in options_map_num_to_display.items():
        if display_str == selected_display_str:
            return num_val
    return None


def get_multiselect_options(group_key_in_one_hot_groups):
    options_for_multiselect = {}
    cols = one_hot_groups.get(group_key_in_one_hot_groups, [])
    valid_cols = [col for col in cols if not col.endswith("_nan")]
    for col_name in valid_cols:
        display_name = col_name.replace(group_key_in_one_hot_groups + "_", "").replace("_", " ")
        display_name = display_name.replace("/herb,", "/herb, ")
        options_for_multiselect[display_name] = col_name
    return options_for_multiselect

# Streamlit UI
st.title("🌱 Plant Recommender 🌱")
st.markdown("Select your desired plant features in the sidebar to get recommendations.")

if df_original is None:
    st.info("Data could not be loaded. Please check the CSV file and console output.")
    st.stop()

st.sidebar.header("Plant Criteria")
user_criteria_streamlit = {}

temp_min_user = get_float_input_st(
    "Max tolerable LOWEST Temperature (°F) (e.g., -20 means tolerates down to -20F or warmer)",
    default_value=None # Allow blank input
)
if temp_min_user is not None:
    user_criteria_streamlit['Temperature_Minimum_F'] = temp_min_user

st.sidebar.subheader("Soil pH Range")
ph_min_user = get_float_input_st("Desired MINIMUM soil pH (e.g., 5.0)", default_value=None)
ph_max_user = get_float_input_st("Desired MAXIMUM soil pH (e.g., 7.0)", default_value=None)

if ph_min_user is not None: user_criteria_streamlit['pH_Minimum'] = ph_min_user
if ph_max_user is not None: user_criteria_streamlit['pH_Maximum'] = ph_max_user

st.sidebar.subheader("Tolerances & Characteristics")
# Define the display options for ordinal features
# The key is the DataFrame column name, value is a tuple: (Display Label, {numerical_value: display_string})
ordinal_feature_ui_map = {
    'Drought_Tolerance': ("Drought Tolerance", {1:"Low", 2:"Medium", 3:"High"}),
    'Shade_Tolerance': ("Shade Tolerance", {0:"Full Sun", 1:"Partial", 2:"Full Shade"}),
    'Salinity_Tolerance': ("Salinity Tolerance", {0:"None", 1:"Low", 2:"Medium", 3:"High"}),
    'Growth_Rate': ("Growth Rate", {1:"Slow", 2:"Moderate", 3:"Rapid"}),
    'Lifespan': ("Lifespan", {1:"Short", 2:"Medium", 3:"Long"}),
    'Toxicity': ("Maximum Acceptable Toxicity", {0:"None", 1:"Minor", 2:"Moderate", 3:"High"})
}

for feature_key, (label, num_to_display_map) in ordinal_feature_ui_map.items():
    selected_value = get_ordinal_selectbox_st(label, options_map_num_to_display=num_to_display_map)
    if selected_value is not None:
        user_criteria_streamlit[feature_key] = selected_value


st.sidebar.subheader("Categorical Features")
gh_options_map = get_multiselect_options("Growth_Habit")
if gh_options_map:
    selected_gh_display = st.sidebar.multiselect("Preferred Growth Habit(s)", options=list(gh_options_map.keys()))
    if selected_gh_display:
        user_criteria_streamlit['Growth_Habit'] = [gh_options_map[name] for name in selected_gh_display]

bp_options_map = get_multiselect_options("Bloom_Period")
if bp_options_map:
    selected_bp_display = st.sidebar.multiselect("Preferred Bloom Period(s)", options=list(bp_options_map.keys()))
    if selected_bp_display:
        user_criteria_streamlit['Bloom_Period'] = [bp_options_map[name] for name in selected_bp_display]

if "Flower_Conspicuous_Yes" in feature_cols:
    flower_conspic = st.sidebar.checkbox("Showy Flowers?", value=False) # Default to False (no specific preference)
    if flower_conspic: user_criteria_streamlit['Flower_Conspicuous'] = ["Flower_Conspicuous_Yes"]

if "Fall_Conspicuous_Yes" in feature_cols:
    fall_conspic = st.sidebar.checkbox("Fall Conspicuous?", value=False)
    if fall_conspic: user_criteria_streamlit['Fall_Conspicuous'] = ["Fall_Conspicuous_Yes"]

# Action Button and Display Results
if st.sidebar.button("🌿 Find Matching Plants 🌿"):
    if not user_criteria_streamlit:
        st.warning("Please select at least one criterion to get recommendations.")
    else:
        st.subheader("Your Selected Criteria for Ideal Profile:")
        for crit_key_display, crit_val in user_criteria_streamlit.items():
            display_key = crit_key_display.replace('_', ' ')
            if isinstance(crit_val, list):
                display_values = [v.split('_')[-1].replace("_", " ") for v in crit_val]
                st.write(f"- **{display_key}**: {', '.join(display_values)}")
            elif crit_key_display in ordinal_feature_ui_map: # Check if it's one of the mapped ordinals
                # Get the display string for the selected numerical value
                display_val_str = ordinal_feature_ui_map[crit_key_display][1].get(crit_val, str(crit_val))
                st.write(f"- **{display_key}**: {display_val_str}")
            else:
                st.write(f"- **{display_key}**: {crit_val}")
        st.markdown("---")

        recommended_plants_df = recommend_plants_knn_ideal_profile(
            user_criteria_streamlit,
            k=NUM_RECOMMENDATIONS_TO_SHOW,
            metric=METRIC_FOR_KNN
        )

        if not recommended_plants_df.empty:
            st.subheader(f"Top {len(recommended_plants_df)} Plant Recommendations:")
            for i, (index, row) in enumerate(recommended_plants_df.iterrows()):
                st.markdown(f"#### {i+1}. {row['Scientific_Name_x']} (ID: {row['id']})")
                with st.expander("Show All Features"):
                    st.write("Key Numeric & Ordinal Features:")
                    key_numerics_ordinals = ['Growth_Rate', 'Lifespan', 'Toxicity', 'Drought_Tolerance',
                                    'Hedge_Tolerance', 'Moisture_Use', 'pH_Minimum', 'pH_Maximum',
                                    'Salinity_Tolerance', 'Shade_Tolerance', 'Temperature_Minimum_F']
                    for kno_col in key_numerics_ordinals:
                        if kno_col in row and pd.notna(row[kno_col]):
                            display_val = row[kno_col]
                            if kno_col in ordinal_feature_ui_map:
                                try:
                                    # Attempt to convert to int for map lookup if it's float
                                    num_val_for_map = int(float(row[kno_col]))
                                    display_val = ordinal_feature_ui_map[kno_col][1].get(num_val_for_map, str(row[kno_col]))
                                except ValueError:
                                    display_val = str(row[kno_col]) # Fallback if not convertible
                            st.write(f"  - **{kno_col.replace('_', ' ')}**: {display_val}")

                    st.write("\nOther Categorical Features (Present):")
                    for group_key_display, cols_in_group in one_hot_groups.items():
                        present_features_in_group = []
                        for col_name_actual in cols_in_group:
                            if col_name_actual in row and row[col_name_actual] == 1:
                                display_feature_name = col_name_actual.replace(group_key_display + "_", "").replace("_", " ")
                                present_features_in_group.append(display_feature_name)
                        if present_features_in_group:
                            st.write(f"  - **{group_key_display.replace('_', ' ')}**: {', '.join(present_features_in_group)}")
                st.markdown("---")
        else:
            st.info("No plants found closely matching your ideal profile based on the selected criteria. Try adjusting your preferences.")
else:
    st.info("Adjust criteria in the sidebar and click 'Find Matching Plants'.")

st.markdown("---")
st.caption("Data Source: Natural Resources Conservation Service. PLANTS Database. United States Department of Agriculture. Accessed May 10, 2025, from https://plants.usda.gov.")