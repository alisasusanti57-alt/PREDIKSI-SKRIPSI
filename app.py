import streamlit as st
import torch
import torch.nn as nn
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ================= CONFIG =================

st.set_page_config(
    page_title="Prediksi Judul Skripsi",
    page_icon="🎓",
    layout="wide"
)



# ================= MODEL =================

class ANN(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim,64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32,1)

        )


    def forward(self,x):

        return self.network(x)




# ================= LOAD MODEL =================


vectorizer = joblib.load(
    "tfidf.pkl"
)


checkpoint = torch.load(
    "ann_model.pth",
    map_location="cpu"
)


model = ANN(
    checkpoint["input_dim"]
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.eval()



# ================= UI =================


st.title("🎓 Prediksi Judul Skripsi")

st.write(
    "Klasifikasi judul skripsi menjadi STEM atau NON-STEM menggunakan ANN"
)



judul = st.text_area(
    "Masukkan Judul Skripsi",
    height=120,
    placeholder="Contoh: Analisis Sentimen Pengguna Mobile Banking Menggunakan Artificial Neural Network"
)




# ================= PREDIKSI =================


if st.button(
    "🚀 Prediksi",
    use_container_width=True
):


    if judul.strip()=="":


        st.warning(
            "Masukkan judul terlebih dahulu"
        )



    else:


        # TF-IDF

        data = vectorizer.transform(
            [judul]
        )


        tensor = torch.tensor(
            data.toarray(),
            dtype=torch.float32
        )



        # Prediksi

        with torch.no_grad():

            output = model(
                tensor
            )

            prob = torch.sigmoid(
                output
            ).item()



        hasil = (
            "STEM"
            if prob >= 0.5
            else
            "NON-STEM"
        )


        confidence = max(
            prob,
            1-prob
        )



        st.divider()



        # ================= HASIL =================


        col1,col2 = st.columns(2)



        with col1:


            st.subheader(
                "📊 Hasil Prediksi"
            )


            if hasil=="STEM":

                st.success(
                    f"🧠 {hasil}"
                )

            else:

                st.error(
                    f"📚 {hasil}"
                )



            st.metric(
                "Probabilitas STEM",
                f"{prob*100:.2f}%"
            )


            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )


            st.progress(
                float(prob)
            )




        # ================= GAUGE =================


        with col2:


            fig = go.Figure(

                go.Indicator(

                    mode="gauge+number",

                    value=prob*100,

                    title={
                        "text":
                        "Probabilitas STEM (%)"
                    },


                    gauge={

                        "axis":{
                            "range":
                            [0,100]
                        }

                    }

                )

            )


            fig.update_layout(
                height=350
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )