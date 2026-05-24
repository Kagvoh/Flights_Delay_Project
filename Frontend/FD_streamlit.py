import streamlit as st
import requests
import joblib

# FastAPI endpoint
API_URL = "http://localhost:8000/predict"

def main():
    st.set_page_config(
        page_title='Flight Delay Prediction',
        page_icon='✈️',
        layout='wide'
    )


    # CUSTOM CSS
    st.markdown(
            """
        <style>

        /* =========================
        MAIN BACKGROUND
        ========================== */

        .stApp {
            background: linear-gradient(
                135deg,
                #F8FAFC,
                #EEF2FF
            );
        }

        /* =========================
        TITLE
        ========================== */

        .title {

            font-size: 56px;
            font-weight: 800;

            color: #111827;

            text-align: center;

            margin-top: 10px;
            margin-bottom: 10px;

            letter-spacing: -1px;
        }

        .subtitle {

            font-size: 21px;

            color: #6B7280;

            text-align: center;

            margin-bottom: 40px;
        }

        /* =========================
        SECTION TITLE
        ========================== */

        h2, h3 {

            color: #111827;

            font-weight: 700;
        }

        /* =========================
        BUTTON
        ========================== */

        .stButton > button {

            width: 100%;

            height: 58px;

            border-radius: 16px;

            background: linear-gradient(
                90deg,
                #4F46E5,
                #2563EB
            );

            color: white;

            font-size: 20px;

            font-weight: 700;

            border: none;

            transition: 0.3s;

            box-shadow:
                0 4px 14px rgba(0,0,0,0.12);
        }

        .stButton > button:hover {

            transform: translateY(-3px);

            box-shadow:
                0 10px 25px rgba(37,99,235,0.25);
        }

        /* =========================
        METRIC CARD
        ========================== */

        div[data-testid="stMetric"] {

            background: white;

            padding: 22px;

            border-radius: 20px;

            border: 1px solid #E5E7EB;

            box-shadow:
                0 4px 12px rgba(0,0,0,0.06);

            text-align: center;
        }

        div[data-testid="stMetricLabel"] {

            justify-content: center;

            font-size: 16px;

            color: #6B7280;

            font-weight: 600;
        }

        div[data-testid="stMetricValue"] {

            justify-content: center;

            font-size: 34px;

            font-weight: 800;

            color: #111827;
        }

        /* =========================
        INPUT BOXES
        ========================== */

        .stSelectbox label,
        .stNumberInput label {

            color: #111827 !important;

            font-weight: 700;

            font-size: 15px;
        }

        
        /* Selectbox */

        .stSelectbox > div > div {

            background-color: white !important;

            border-radius: 14px !important;

            border: 1px solid #D1D5DB !important;
        }

        /* Number Input FULL BOX */

        .stNumberInput > div {

            background-color: white !important;

            border-radius: 14px !important;

            border: 1px solid #D1D5DB !important;

            padding: 2px;
        }

        /* Actual input */

        .stNumberInput input {

            background-color: white !important;

            border: none !important;

            box-shadow: none !important;

            color: #111827 !important;

            font-weight: 500;
        }

        /* Plus Minus buttons */

        .stNumberInput button {

            background-color: white !important;

            border: none !important;

            color: #374151 !important;
        }
        /* =========================
        INFO / SUCCESS / ERROR
        ========================== */

        .stAlert {

            border-radius: 18px;
        }

        /* =========================
        DATAFRAME
        ========================== */

        .stDataFrame {

            border-radius: 18px;

            overflow: hidden;

            border: 1px solid #E5E7EB;
        }

        /* =========================
        TAB STYLE
        ========================== */

        button[data-baseweb="tab"] {

            font-size: 17px;

            font-weight: 700;

            color: #6B7280;

            padding: 10px 24px;
        }

        button[data-baseweb="tab"][aria-selected="true"] {

            color: #2563EB;
        }

        /* =========================
        SIDEBAR
        ========================== */

        section[data-testid="stSidebar"] {

            background: white;

            border-right: 1px solid #E5E7EB;
        }

        /* =========================
        DIVIDER
        ========================== */

        hr {

            border: none;

            height: 1px;

            background: #E5E7EB;
        }
        div[data-testid="stMetricValue"] {

            white-space: normal !important;

            overflow-wrap: break-word !important;

            line-height: 1.2;

            font-size: 28px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    # TITLE
    st.markdown('<div class="title">✈️ Flight Delay Prediction System</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="subtitle">Predict flight delays using Machine Learning Models</div>',
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs([
        '📊 Airline Analytics',
        '✈️ Flight Prediction'
    ])
    
    # Upload Origin, dest and airline from data
    origin_list = joblib.load('../backend/models/origin_list.pkl')

    dest_list = joblib.load('../backend/models/dest_list.pkl')

    airline_list = joblib.load('../backend/models/airline_list.pkl')
    
    airline_stats = joblib.load( '../backend/models/airline_stats.pkl')
    
    with tab1:
        st.markdown(
            '## Airline Delay Analytics'
        )
        
        st.info(
            '📊 Statistics are generated from U.S. airline flight data in 2008.'
        )
        
        st.markdown('### Top Reliable Airlines')
        top_airlines = airline_stats.sort_values('DelayRate').head(5)
        st.dataframe(top_airlines)
            
        top1, top2, top3 = st.columns(3)
        with top1:
            st.metric(
                'Average Delay Rate',
                f"{airline_stats['DelayRate'].mean():.1f}%"
            )
            
        best_airline = airline_stats.sort_values('DelayRate').iloc[0]
        with top2:
            st.metric(
                'Most Reliable Airline',
                best_airline.name
            )
            
        worst_airline = airline_stats.sort_values('DelayRate',ascending=False).iloc[0]    
        with top3:
            st.metric(
                'Highest Delay Airline',
                worst_airline.name
            )
            
        st.markdown('### Delay Rate by Airline')   
        chart_df = airline_stats.sort_values('DelayRate', ascending=False)
        st.bar_chart(chart_df['DelayRate'])
        
        selected_airline = st.selectbox('Choose Airline',airline_stats.index.tolist())
        info = airline_stats.loc[selected_airline]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                'Delay Rate',
                f"{info['DelayRate']:.1f}%"
            )
        
        with c2:
            st.metric(
                'Reliability',
                info['Reliability']
            )
            
        with c3:
            st.metric(
                'Average Distance',
                f"{info['AvgDistance']:.0f} miles"
            )
            
    with tab2:   
        left_col, right_col = st.columns([1,1])     
        result = None
        with left_col: 
            # MODEL SELECT
            model_name = st.selectbox(
                'Choose Model',
                [
                    'Logistic Regression',
                    'Decision Tree',
                    'XGBoost'
                ]
            )

            st.divider()

            # INPUT ROWS
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                month = st.selectbox('Month', list(range(1, 13)))

            with col2:
                day = st.selectbox('Day Of Month', list(range(1,32)))

            with col3:
                dayofweek = st.selectbox('Day Of Week', [1,2,3,4,5,6,7])

            with col4:
                dep_time = st.number_input(
                    'CRS Departure Time',0,2359,0
                )
            
            with col5:
                arr_time = st.number_input(
                    'CRS Arrival Time',0,2359,0
                )

            # SECOND ROW
            col6, col7, col8 = st.columns(3)

            with col6:
                carrier = st.selectbox('Airline', airline_list)

            with col7:
                origin = st.selectbox('Origin Airport', origin_list)

            with col8:
                dest = st.selectbox('Destination Airport', dest_list)

            st.divider()

            result = None
            # BUTTON click to predict
            if st.button('🚀 Predict Delay'):

                #Load data for predicting
                payload = {
                    'Month': month,
                    'DayofMonth': day,
                    'DayOfWeek': dayofweek,
                    'CRSDepTime': dep_time,
                    'CRSArrTime' : arr_time,
                    'UniqueCarrier': carrier,
                    'Origin': origin,
                    'Dest': dest,
                    'model_name': model_name
                }

                # Call FastAPI endpoint and get prediction result
                response = requests.post(
                    API_URL,
                    json = payload
                )

                if response.status_code != 200:

                    st.error('Backend Error')

                    st.write(response.text)

                    st.stop()

                result = response.json()
        
        with right_col:
            if result:
                # Display prediction result
                st.markdown('## Prediction Result')
                
                prob = result['probability_delay'] * 100

                if result['prediction'] == 1:
                    st.error('⚠️ Flight likely DELAYED')
                else:
                    st.success('✅ Flight likely ON TIME')
                
                day_name_map = {
                    1: 'Monday',
                    2: 'Tuesday',
                    3: 'Wednesday',
                    4: 'Thursday',
                    5: 'Friday',
                    6: 'Saturday',
                    7: 'Sunday'
                }
                
                arr_time_str = f"{arr_time:04d}"

                formatted_arr_time = (
                    f"{arr_time_str[:2]}:{arr_time_str[2:]}"
                )
                
                dep_time_str = f"{dep_time:04d}"

                formatted_dep_time = (
                    f"{dep_time_str[:2]}:{dep_time_str[2:]}"
                )
                
                st.markdown('### ✈️ Flight Information')

                inf_col1, inf_col2 = st.columns(2)
                
                with inf_col1:
                    st.info(
                    f"""
                    📅 Flight Date: {month}/{day}

                    🗓 Day Of Week: {day_name_map[dayofweek]}
                    
                    🌍 Route: {origin} → {dest}
                    
                    ✈ Airline: {carrier}
                    """
                    )
                    
                with inf_col2:
                    st.info(
                    f"""
                    🛬 CRS Departure Time:
                        {formatted_dep_time}
                                
                    🛬 CRS Arrival Time:
                        {formatted_arr_time}

                    🕒 Flight Duration:
                        {result['CRS_elapsed_time']:.0f} minutes
                    
                    📍 Distance:
                        {result['distance']:.0f} miles
                    """
                    )

                top1, top2 = st.columns(2)
                with top1:
                    st.metric(
                        'Delay Probability',
                        f"{prob:.2f}%"
                    )
                    
                with top2:
                    if prob < 40:
                        st.metric(
                            "Delay Risk", "Low"
                        )

                    elif prob < 70:
                        st.metric(
                            "Delay Risk", "Moderate"
                        )

                    else:
                        st.metric(
                            "Delay Risk","High"
                        )

                bottom1, bottom2 = st.columns(2)
                
                with bottom1:
                    st.metric(
                        'CRS Departure Period',
                        result['CRS_departure_period']
                    )
                    
                with bottom2:
                    st.metric(
                        'CRS Arrival Period',
                        result['CRS_arrival_period']
                    )
            
if __name__ == "__main__":
    main()          
