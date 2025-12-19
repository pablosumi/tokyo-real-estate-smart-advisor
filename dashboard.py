import streamlit as st
import numpy as np
import pandas as pd
from src.inference import make_prediction

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Tokyo Real Estate Advisor",
    page_icon="🏠",
    layout="wide"
)

def main():
    st.title("Tokyo Real Estate Smart Advisor")
    st.caption("Estimate property market value using MLIT historical transactions")

    with st.form("valuation_form"):
        tab1, tab2, tab3 = st.tabs(["Context & Usage", "Building Specs", "Land & Zoning"])

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
                prop_type = st.selectbox("Type", options=['Pre-owned Condominiums, etc.',
                                                          'Residential Land(Land and Building)'])
                
                floor_plan = st.selectbox("Floor Plan", options=[None, '1K', '3DK', '1R', '2LDK', '1LDK', '1DK', '3LDK', '2DK',
                                                                'Open Floor', '4LDK', '1LDK+S', '2K', '1K+S', 'Studio Apartment',
                                                                '3LK', 'Duplex', '3LDK+S', '2LDK+S', '4LDK+S', '3K', '5LDK',
                                                                '3DK+S', '4DK', '2DK+S', '6LDK', '2LK', '7LDK', '1DK+S', '1LK',
                                                                '3LD', '1R+S', '4K', '4DK+S', '2LK+S', '2LD+S', '3LD+S', '2K+S',
                                                                '5LDK+S', '2LD', '5DK', '1L+S', '6LDK+S', '3LDK+K', '1L', '6DK',
                                                                '1LK+S', '8LDK', '5LK', '5K', '6DK+S', '7LDK+S', '3K+S', '7DK',
                                                                '6K', '1LDK+K', '5K+S', '5DK+S', '3LK+S', '4K+S', '8LDK+S', '2L+S',
                                                                '4LK', '6K+S', '1LD+S', '6LK', '4L+K'])

            with col2:
                building_year = st.number_input("Building Construction Year", min_value=1945, max_value=2030, value=2010)

                region = st.selectbox("Region", options=[None, 'Commercial Area', 'Residential Area',
                                                         'Industrial Area', 'Potential Residential Area'])
                
                area = st.number_input("Area (m²)", min_value=1.0, value=65.0)


        with tab2:
            col1, col2 = st.columns(2)
            with col1:

                total_floor_area = st.number_input("Total Floor Area (m²)", min_value=0.0, value=95.0)

            with col2:
                structure = st.selectbox("Structure", options=['RC', 'SRC', None, 'W', 'S', 'RC, W', 'RC, S', 'SRC, RC', 'LS',
                                                                'S, W', 'B', 'RC, W, B', 'W, LS', 'RC, S, W', 'RC, LS', 'SRC, W',
                                                                'S, B', 'SRC, S', 'W, B', 'B, LS', 'S, W, LS', 'RC, B', 'S, LS',
                                                                'S, W, B', 'RC, S, LS'])

        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                land_shape = st.selectbox("Land Shape", options=[None, 'Irregular Shaped', 'Semi-rectangular Shaped',
                                                                'Rectangular Shaped', 'Trapezoidal Shaped', 'Semi-square Shaped',
                                                                'Semi-trapezoidal Shaped', 'Square Shaped', 'Semi-shaped',
                                                                '&quot;Flag-shaped&quot; etc.'])
                frontage = st.number_input("Frontage (m)", min_value=0.0, max_value=50.0, value=10.0)
                breadth = st.number_input("Road Breadth (m)", min_value=0.0, max_value=100.0, value=0.0)
                road_direction = st.selectbox("Road Direction", options=[None, 'East', 'No facing road', 'North', 'Northeast',
                                                                        'Northwest', 'South', 'Southeast', 'Southwest', 'West'])
            with col2:
                classification = st.selectbox("Road Classification", options=[None, 'Access Road', 'Agricultural Road', 'City Road', 'Forest Road',
                                                                            'Hokkaido Prefectural Road', 'Kyoto/ Osaka Prefectural Road',
                                                                            'National Road', 'Prefectural Road', 'Private Road', 'Public Road',
                                                                            'Road', 'Tokyo Metropolitan Road', 'Town Road', 'Village Road',
                                                                            'Ward Road'])
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
        'FloorAreaRatio': floor_area_ratio
    }

    if submit:
        try:
            with st.spinner("Processing..."):
                predicted_price = make_prediction(user_input)
            st.success("Estimation complete")
            st.metric(label="Predicted Market Value", value=f"¥{predicted_price:,.0f}")
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
