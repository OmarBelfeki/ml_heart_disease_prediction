import joblib
import flet as ft
import pandas as pd

data = joblib.load("model_")


def main(page: ft.Page) -> None:
    page.window_width = 400
    page.window_height = 750
    page.title = "ML: Heart Disease Prediction"
    page.scroll = ft.ScrollMode.ALWAYS

    def work(e: ft.ControlEvent) -> None:
        new_data = pd.DataFrame(
            data={
                'age': int(age.value),
                'sex': int(sex.value),
                'cp': int(cp.value),
                'trestbps': int(trestbps.value),
                'chol': int(chol.value),
                'fbs': int(fbs.value),
                'restecg': int(restecg.value),
                'thalach': int(thalach.value),
                'exang': int(exang.value),
                'oldpeak': int(oldpeak.value),
                'slope': int(slope.value),
                'ca': int(ca.value),
                'thal': int(thal.value)
            },
            index=[0]
        )

        target.value = "No Disease" if data.predict(new_data)[0] == 0 else "Disease"
        page.update()

    age = ft.TextField(label="Age")
    sex = ft.TextField(label="Sex")
    cp = ft.TextField(label="chest pain type (4 values)")
    trestbps = ft.TextField(label="resting blood pressure")
    chol = ft.TextField(label="serum cholestoral in mg/dl")
    fbs = ft.TextField(label="fasting blood sugar > 120 mg/dl")
    restecg = ft.TextField(label="resting electrocardiographic results (values 0,1,2)")
    thalach = ft.TextField(label="maximum heart rate achieved")
    exang = ft.TextField(label="exercise induced angina")
    oldpeak = ft.TextField(label="oldpeak = ST depression induced by exercise relative to rest")
    slope = ft.TextField(label="the slope of the peak exercise ST segment")
    ca = ft.TextField(label="number of major vessels (0-3) colored by flourosopy")
    thal = ft.TextField(label="thal: 0 = normal; 1 = fixed defect; 2 = reversable defect")

    btn = ft.ElevatedButton(text="Predict", width=400, bgcolor=ft.colors.GREY, on_click=work)

    target = ft.Text(color=ft.colors.GREEN, size=30, width=400, text_align=ft.TextAlign.CENTER)

    page.add(
        ft.Column(
            controls=[
                age,
                sex,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal,
                btn,
                target
            ]
        )
    )


ft.app(target=main)
