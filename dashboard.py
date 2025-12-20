import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from src.inference import make_prediction
from src.chat import get_chat_completion
import altair as alt
from datetime import date

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Tokyo Real Estate Advisor",
    page_icon="📈",
    layout="wide"
)

DATA_PATH = Path(__file__).resolve().parent / "data" / "tokyo-clean.parquet"


@st.cache_data
def load_transaction_data() -> pd.DataFrame:
    return pd.read_parquet(
        DATA_PATH,
        columns=["Municipality", "FloorPlan", "TransactionYear", "TradePriceYen"]
    )

STRUCTURE_LABELS = {
    'RC': 'Reinforced Concrete',
    'SRC': 'Steel Reinforced Concrete',
    'W': 'Wood',
    'S': 'Steel Frame',
    'LS': 'Light Gauge Steel',
    'B': 'Concrete Block',
    'RC, W': 'RC & Wood mix',
    'RC, S': 'RC & Steel mix',
    'SRC, RC': 'SRC & RC mix',
    'S, W': 'Steel & Wood mix',
    'RC, W, B': 'RC, Wood, & Block',
    'W, LS': 'Wood & Light Steel',
    'RC, S, W': 'RC, Steel, & Wood',
    'RC, LS': 'RC & Light Steel',
    'SRC, W': 'SRC & Wood',
    'S, B': 'Steel & Block',
    'SRC, S': 'SRC & Steel',
    'W, B': 'Wood & Block',
    'B, LS': 'Block & Light Steel',
    'S, W, LS': 'Steel, Wood, & Light Steel',
    'RC, B': 'RC & Block',
    'S, LS': 'Steel & Light Steel',
    'S, W, B': 'Steel, Wood, & Block',
    'RC, S, LS': 'RC, Steel, & Light Steel',
    None: 'Unknown'
}

ORDERED_FLOOR_PLANS = [
    None,
    # 1 Room / Studios
    '1R', '1R+S', '1K', '1K+S', '1DK', '1DK+S', '1L', '1L+S', '1LD+S', '1LK', '1LK+S', '1LDK', '1LDK+K', '1LDK+S',
    # 2 Rooms
    '2K', '2K+S', '2DK', '2DK+S', '2LD', '2LD+S', '2LK', '2LK+S', '2L+S', '2LDK', '2LDK+S',
    # 3 Rooms
    '3K', '3K+S', '3DK', '3DK+S', '3LD', '3LD+S', '3LK', '3LK+S', '3LDK', '3LDK+K', '3LDK+S',
    # 4 Rooms
    '4K', '4K+S', '4DK', '4DK+S', '4LK', '4L+K', '4LDK', '4LDK+S',
    # 5 Rooms
    '5K', '5K+S', '5DK', '5DK+S', '5LK', '5LDK', '5LDK+S',
    # 6+ Rooms
    '6K', '6K+S', '6DK', '6DK+S', '6LK', '6LDK', '6LDK+S',
    '7DK', '7LDK', '7LDK+S', '8LDK', '8LDK+S',
    # Special Types
    'Studio Apartment', 'Open Floor', 'Duplex'
]

def main():
    st.title("Tokyo Real Estate Smart Advisor")
    #st.caption("Estimate property market value using MLIT historical transactions")

    # CSS for chatbox UI
    st.markdown("""
            <style>
                /* 1. Hide all chat icons/avatars */
                [data-testid="stChatMessageAvatarUser"],
                [data-testid="stChatMessageAvatarAssistant"] {
                    display: none;
                }

                /* 2. Reset padding for all messages now that icons are gone */
                [data-testid="stChatMessage"] {
                    padding-left: 1rem;
                    padding-right: 1rem;
                }

                /* 3. Right-align the User message container */
                [data-testid="stChatMessageUser"] {
                    display: flex;
                    flex-direction: row-reverse;
                    text-align: right;
                    background-color: transparent; /* Optional: makes it look cleaner */
                }

                /* 4. Ensure the text inside the user message takes up only as much space as needed */
                [data-testid="stChatMessageUser"] .stChatMessageContent {
                    margin-left: auto;
                    background-color: #f0f2f6; /* Light grey bubble for user */
                    border-radius: 15px;
                    padding: 10px 15px;
                }

                /* 5. Assistant message styling (Left-aligned) */
                [data-testid="stChatMessageAssistant"] .stChatMessageContent {
                    margin-right: auto;
                    background-color: #e1f5fe; /* Soft blue bubble for assistant */
                    border-radius: 15px;
                    padding: 10px 15px;
                }
            </style>
        """, unsafe_allow_html=True)
    
    

    transactions = load_transaction_data()
    price_chart_container = st.container()
    result_placeholder = st.container()

    with st.form("valuation_form"):
        tab1, tab2 = st.tabs(["Basic Info", "Advanced Specs"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                municipality = st.selectbox("Municipality", options=[
                        # 東京23区（Wards, fixed, conventional order）
                        '千代田区 (Chiyoda Ward)', '中央区 (Chuo Ward)', '港区 (Minato Ward)',
                        '新宿区 (Shinjuku Ward)', '文京区 (Bunkyo Ward)', '台東区 (Taito Ward)',
                        '墨田区 (Sumida Ward)', '江東区 (Koto Ward)', '品川区 (Shinagawa Ward)',
                        '目黒区 (Meguro Ward)', '大田区 (Ota Ward)', '世田谷区 (Setagaya Ward)',
                        '渋谷区 (Shibuya Ward)', '中野区 (Nakano Ward)', '杉並区 (Suginami Ward)',
                        '豊島区 (Toshima Ward)', '北区 (Kita Ward)', '荒川区 (Arakawa Ward)',
                        '板橋区 (Itabashi Ward)', '練馬区 (Nerima Ward)', '足立区 (Adachi Ward)',
                        '葛飾区 (Katsushika Ward)', '江戸川区 (Edogawa Ward)',

                        # 市 (cities)
                        '昭島市 (Akishima City)', 'あきる野市 (Akiruno City)',
                        '青梅市 (Oume City)', '小金井市 (Koganei City)',
                        '国分寺市 (Kokubunji City)', '国立市 (Kunitachi City)',
                        '小平市 (Kodaira City)', '狛江市 (Komae City)',
                        '清瀬市 (Kiyose City)', '調布市 (Chofu City)',
                        '立川市 (Tachikawa City)', '多摩市 (Tama City)',
                        '西東京市 (Nishitokyo City)', '八王子市 (Hachioji City)',
                        '羽村市 (Hamura City)', '東久留米市 (Higashikurume City)',
                        '東村山市 (Higashimurayama City)', '東大和市 (Higashiyamato City)',
                        '日野市 (Hino City)', '府中市 (Fuchu City)',
                        '福生市 (Fussa City)', '町田市 (Machida City)',
                        '三鷹市 (Mitaka City)', '武蔵野市 (Musashino City)',
                        '武蔵村山市 (Musashimurayama City)', '稲城市 (Inagi City)',

                        # 町・村 (towns & villages)
                        '瑞穂町 (Mizuho Town, Nishitama County)',
                        '日の出町 (Hinode Town, Nishitama County)',
                        '檜原村 (Hinohara Village, Nishitama County)',
                        '奥多摩町 (Okutama Town, Nishitama County)',

                        # 島しょ部 (islands)
                        '大島町 (Oshima Town)', '神津島村 (Kozushima Village)',
                        '新島村 (Niijima Village)', '三宅村 (Miyake Village)',
                        '八丈町 (Hachijo Town)', '小笠原村 (Ogasawara Village)'
                    ])
                
                floor_plan = st.selectbox("Floor Plan", options=ORDERED_FLOOR_PLANS, index=12)

            with col2:
                building_year = st.number_input("Building Construction Year", min_value=1945, max_value=2030, value=2003)
                
                area = st.number_input("Area (m²)", min_value=1.0, value=40.0)


        with tab2:
            col1, col2 = st.columns(2)
            with col1:

                prop_type = st.selectbox("Type", options=[None, 'Pre-owned Condominiums, etc.', 'Residential Land(Land and Building)'])

                region = st.selectbox("Region", options=[None, 'Commercial Area', 'Industrial Area', 'Potential Residential Area', 'Residential Area'])

                structure = st.selectbox("Structure", options=list(STRUCTURE_LABELS.keys()), index=24, format_func=lambda x: STRUCTURE_LABELS.get(x, x), help="RC and SRC are common for apartments; W is common for houses."
)
                
                land_shape = st.selectbox("Land Shape", options=[None, 'Irregular Shaped','Rectangular Shaped','Semi-rectangular Shaped','Semi-shaped',
                                                                'Semi-square Shaped', 'Semi-trapezoidal Shaped', 'Square Shaped', 'Trapezoidal Shaped'])
                
                road_direction = st.selectbox("Road Direction", options=[None, 'No facing road', 'East', 'North', 'Northeast',
                                                                        'Northwest', 'South', 'Southeast', 'Southwest', 'West'])
                
                classification = st.selectbox("Road Classification", options=[None, 'Access Road', 'Agricultural Road', 'City Road', 'Forest Road',
                                                                            'Hokkaido Prefectural Road', 'Kyoto/ Osaka Prefectural Road',
                                                                            'National Road', 'Prefectural Road', 'Private Road', 'Public Road',
                                                                            'Road', 'Tokyo Metropolitan Road', 'Town Road', 'Village Road',
                                                                            'Ward Road'])

            with col2:
                
                total_floor_area = st.number_input("Total Floor Area (m²)", min_value=1.0, value=200.0)
                frontage = st.number_input("Frontage (m)", min_value=0.0, max_value=50.0, value=7.5)
                breadth = st.number_input("Road Breadth (m)", min_value=0.0, max_value=100.0, value=8.0)
                coverage_ratio = st.slider("Coverage Ratio (%)", 0, 500, 60)
                floor_area_ratio = st.slider("Floor Area Ratio (%)", 0, 1300, 200)


        st.divider()
        submit = st.form_submit_button("Calculate Estimate", type="primary", use_container_width=True)

    user_input = {
        'Type': prop_type,
        'Region': region,
        'Municipality': municipality,
        'FloorPlan': floor_plan,
        'Area': area,
        'LandShape': land_shape,
        'Frontage': frontage,
        'TotalFloorArea': total_floor_area,
        'BuildingYear': building_year,
        'Structure': structure,
        'RoadDirection': road_direction,
        'Classification': classification,
        'Breadth': breadth,
        'CoverageRatio': coverage_ratio,
        'FloorAreaRatio': floor_area_ratio,
        'TransactionYear': date.today().year
    }

    print("columnszzz:",transactions.columns)

    median_price = (
        transactions[(transactions["Municipality"] == municipality) & (transactions["FloorPlan"] == floor_plan)]
        .groupby("TransactionYear", as_index=False)["TradePriceYen"]
        .median()
        .sort_values("TransactionYear")
    )

    with price_chart_container:
            # Create two columns with equal width (1:1 ratio)
            col_chart, col_chat = st.columns([1, 1])

            with col_chart:
                st.subheader(f"Median Transaction Price By Year\n {municipality}, {floor_plan}")
                if not median_price.empty:
                    hover = alt.selection_point(
                        fields=["TransactionYear"],
                        nearest=True,
                        on="mouseover",
                        empty=False,
                    )

                    base = alt.Chart(median_price).encode(
                        x=alt.X("TransactionYear:Q", axis=alt.Axis(format="d", labelAngle=0), title=None),
                        y=alt.Y(
                            "TradePriceYen:Q", 
                            axis=alt.Axis(
                                # This expression prepends ¥ and formats the number with commas
                                labelExpr="'¥' + format(datum.value, ',')", 
                                grid=True
                            ),
                            title=None
                        )
                    )

                    line = base.mark_line(color='#1E90FF') # Using your blue theme color
                    points = base.mark_point(filled=True, size=100, color='#1E90FF').encode(
                        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
                        tooltip=[
                            alt.Tooltip("TransactionYear:Q", title="Year", format="d"),
                            alt.Tooltip("TradePriceYen:Q", title="Price", format=",")
                        ]
                    ).add_params(hover)

                    st.altair_chart(line + points, use_container_width=True)
                else:
                    st.info("No transaction data available for this selection.")

            with col_chat:
                st.subheader("AI Advisor")
                
                # 1. Initialize chat history in session state if it doesn't exist
                if "messages" not in st.session_state:
                    st.session_state.messages = [
                        {"role": "assistant", "content": f"Ready when you are. What's on your radar?"}
                    ]

                # 2. Chat History Container (Fixed height with scrolling)
                chat_container = st.container(height=350)
                
                with chat_container:
                    for msg in st.session_state.messages:
                        st.chat_message(msg["role"]).write(msg["content"])

                # 3. Chat Input
                prompt = st.chat_input(placeholder=f"Ask a question about {municipality}...")

                if prompt:
                    st.session_state.messages.append({"role": "user", "content": prompt})

                    # Inline loading indicator inside the chat box (no global overlay)
                    with chat_container:
                        loading_placeholder = st.chat_message("assistant")
                        loading_placeholder.markdown("Thinking...")

                    try:
                        assistant_reply = get_chat_completion(
                            history=st.session_state.messages,
                            property_context=user_input,
                        )
                    except Exception as exc:
                        assistant_reply = "Sorry, I could not reach the AI advisor just now. Please try again."
                        st.error(f"Chat error: {exc}")

                    # Replace loading indicator with the real response
                    loading_placeholder.empty()
                    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
                    st.rerun()

    if submit:
        try:
            predicted_price = make_prediction(user_input)
            
            with result_placeholder:
                st.success(f"Predicted Market Value: ¥{predicted_price:,.0f}")
                #st.metric(label="", value=f"¥{predicted_price:,.0f}")
                #st.divider() # Optional: visually separates result from the form
                
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
