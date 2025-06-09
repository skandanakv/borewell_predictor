# # # # from flask import Flask, render_template, request
# # # # import numpy as np
# # # # import joblib
# # # # import pandas as pd

# # # # app = Flask(__name__)

# # # # # Load model
# # # # model = joblib.load('groundwater_model.pkl')

# # # # # Feature order (must match training)
# # # # features = [
# # # #     'Annual Replenishable Ground Water Resource',
# # # #     'Natural Discharges During Non-Monsoon Season',
# # # #     'Net Annual Ground Water Availability',
# # # #     'Annual Ground Water Draft (Irrigation)',
# # # #     'Annual Ground Water Draft (Domestic & Industrial)',
# # # #     'Net Ground Water Availability for future use'
# # # # ]

# # # # @app.route('/', methods=['GET', 'POST'])
# # # # def index():
# # # #     prediction = None
# # # #     decision = None

# # # #     if request.method == 'POST':
# # # #         try:
# # # #             input_data = [float(request.form[feature]) for feature in features]
# # # #             X_input = np.array(input_data).reshape(1, -1)
# # # #             predicted_extraction = model.predict(X_input)[0]
# # # #             predicted_extraction = max(predicted_extraction, 0)  # Avoid negative

# # # #             # Borewell decision logic (conservative)
# # # #             if predicted_extraction < 500:
# # # #                 decision = "✅ Borewell CAN be installed at this location."
# # # #             elif predicted_extraction < 1000:
# # # #                 decision = "⚠️ Borewell MAY be installed — needs local approval."
# # # #             else:
# # # #                 decision = "🚫 Borewell CANNOT be installed — groundwater stress too high."

# # # #             prediction = f"{predicted_extraction:.2f}"

# # # #         except Exception as e:
# # # #             prediction = f"Error: {e}"
# # # #             decision = "❌ Invalid input. Please enter numeric values."

# # # #     return render_template('index.html', features=features, prediction=prediction, decision=decision)

   
# # # # @app.route('/dashboard')
# # # # def dashboard():
# # # #     df = pd.read_csv('data/borewell_dataset.csv')

# # # #     # Convert to list of dicts for JSON
# # # #     chart_data = df.to_dict(orient='records')
# # # #     return render_template('dashboard.html', chart_data=chart_data)


# # # # if __name__ == '__main__':
# # # #     app.run(debug=True, port=5050)




# # # from flask import Flask, render_template, request
# # # import numpy as np
# # # import joblib
# # # import pandas as pd

# # # app = Flask(__name__)

# # # # Load model
# # # model = joblib.load('groundwater_model.pkl')

# # # # Features used in training
# # # features = [
# # #     'Total Annual Ground Water Recharge',
# # #     'Total Natural Discharges',
# # #     'Annual Extractable Ground Water Resource',
# # #     'Current Annual Ground Water Extraction For Irrigation',
# # #     'Current Annual Ground Water Extraction For Domestic & Industrial Use',
# # #     'Net Ground Water Availability for future use'
# # # ]

# # # @app.route('/', methods=['GET', 'POST'])
# # # def index():
# # #     prediction = None
# # #     decision = None
# # #     chart_data = None

# # #     if request.method == 'POST':
# # #         try:
# # #             input_data = [float(request.form[feature]) for feature in features]
# # #             X_input = np.array(input_data).reshape(1, -1)
# # #             predicted_extraction = model.predict(X_input)[0]
# # #             predicted_extraction = max(predicted_extraction, 0)

# # #             # Decision
# # #             if predicted_extraction < 500:
# # #                 decision = "✅ Borewell CAN be installed at this location."
# # #             elif predicted_extraction < 1000:
# # #                 decision = "⚠️ Borewell MAY be installed — needs local approval."
# # #             else:
# # #                 decision = "🚫 Borewell CANNOT be installed — groundwater stress too high."

# # #             prediction = f"{predicted_extraction:.2f}"

# # #             # Prepare chart data
# # #             chart_data = {
# # #                 'Predicted Extraction': float(predicted_extraction),
# # #                 'Available Groundwater': float(request.form.get('Annual Extractable Ground Water Resource', 0)),
# # #                 'Irrigation Use': float(request.form.get('Current Annual Ground Water Extraction For Irrigation', 0)),
# # #                 'Domestic Use': float(request.form.get('Current Annual Ground Water Extraction For Domestic & Industrial Use', 0))
# # #             }

# # #         except Exception as e:
# # #             prediction = f"Error: {e}"
# # #             decision = "❌ Invalid input. Please enter numeric values."

# # #     return render_template(
# # #         'index.html',
# # #         features=features,
# # #         prediction=prediction,
# # #         decision=decision,
# # #         chart_data=chart_data
# # #     )

# # # if __name__ == '__main__':
# # #     app.run(debug=True, port=5051)





# # from flask import Flask, render_template, request
# # import numpy as np
# # import joblib
# # import pandas as pd
# # import matplotlib
# # matplotlib.use('Agg')  # Must be BEFORE importing pyplot
# # import matplotlib.pyplot as plt

# # import os

# # app = Flask(__name__)

# # # Load model
# # model = joblib.load('groundwater_model.pkl')

# # # Features used in training
# # features = [
# #     'Total Annual Ground Water Recharge',
# #     'Total Natural Discharges',
# #     'Annual Extractable Ground Water Resource',
# #     'Current Annual Ground Water Extraction For Irrigation',
# #     'Current Annual Ground Water Extraction For Domestic & Industrial Use',
# #     'Net Ground Water Availability for future use'
# # ]

# # @app.route('/', methods=['GET', 'POST'])
# # def index():
# #     prediction = None
# #     decision = None
# #     chart_data = None
# #     plot_path = None

# #     if request.method == 'POST':
# #         try:
# #             input_data = [float(request.form[feature]) for feature in features]
# #             X_input = np.array(input_data).reshape(1, -1)
# #             predicted_extraction = model.predict(X_input)[0]
# #             predicted_extraction = max(predicted_extraction, 0)

# #             # Decision
# #             if predicted_extraction < 500:
# #                 decision = "✅ Borewell CAN be installed at this location."
# #             elif predicted_extraction < 1000:
# #                 decision = "⚠️ Borewell MAY be installed — needs local approval."
# #             else:
# #                 decision = "🚫 Borewell CANNOT be installed — groundwater stress too high."

# #             prediction = f"{predicted_extraction:.2f}"

# #             # Prepare chart data
# #             chart_data = {
# #                 'Predicted Extraction': float(predicted_extraction),
# #                 'Available Groundwater': float(request.form.get('Annual Extractable Ground Water Resource', 0)),
# #                 'Irrigation Use': float(request.form.get('Current Annual Ground Water Extraction For Irrigation', 0)),
# #                 'Domestic Use': float(request.form.get('Current Annual Ground Water Extraction For Domestic & Industrial Use', 0))
# #             }

# #             # --- Plot: Regression line and prediction point ---
# #             x_vals = list(range(1, 8))
# #             y_vals = [i * predicted_extraction / 7 for i in x_vals]

# #             plt.figure(figsize=(5, 3))
# #             plt.plot(x_vals, y_vals, label="Regression Line", color='blue')
# #             plt.scatter([4], [predicted_extraction], color='red', label="Predicted Value", zorder=5)
# #             plt.title("Linear Regression Prediction")
# #             plt.xlabel("Sample Index")
# #             plt.ylabel("Groundwater Extraction")
# #             plt.legend()
# #             plt.tight_layout()

# #             plot_path = "static/plot.png"
# #             if not os.path.exists("static"):
# #                 os.makedirs("static")
# #             plt.savefig(plot_path)
# #             plt.close()

# #         except Exception as e:
# #             prediction = f"Error: {e}"
# #             decision = "❌ Invalid input. Please enter numeric values."

# #     return render_template(
# #         'index.html',
# #         features=features,
# #         prediction=prediction,
# #         decision=decision,
# #         chart_data=chart_data,
# #         plot_path=plot_path
# #     )

# # if __name__ == '__main__':
# #     app.run(debug=True, port=5051)








# from flask import Flask, render_template, request
# import numpy as np
# import joblib
# import os
# import matplotlib

# matplotlib.use('Agg')  # ✅ BEFORE importing pyplot
# import matplotlib.pyplot as plt

# app = Flask(__name__)

# # Load model
# model = joblib.load('groundwater_model.pkl')

# # Features used in training
# features = [
#     'Total Annual Ground Water Recharge',
#     'Total Natural Discharges',
#     'Annual Extractable Ground Water Resource',
#     'Current Annual Ground Water Extraction For Irrigation',
#     'Current Annual Ground Water Extraction For Domestic & Industrial Use',
#     'Net Ground Water Availability for future use'
# ]

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     prediction = None
#     decision = None
#     chart_data = None
#     plot_path = None

#     if request.method == 'POST':
#         try:
#             # ✅ 1. Input processing
#             input_data = [float(request.form[feature]) for feature in features]
#             X_input = np.array(input_data).reshape(1, -1)

#             # ✅ 2. Model prediction
#             predicted_extraction = model.predict(X_input)[0]
#             predicted_extraction = max(predicted_extraction, 0)  # No negative values

#             # ✅ 3. Decision logic
#             if predicted_extraction < 500:
#                 decision = "✅ Borewell CAN be installed at this location."
#             elif predicted_extraction < 1000:
#                 decision = "⚠️ Borewell MAY be installed — needs local approval."
#             else:
#                 decision = "🚫 Borewell CANNOT be installed — groundwater stress too high."

#             prediction = f"{predicted_extraction:.2f}"

#             # ✅ 4. Chart data
#             chart_data = {
#                 'Predicted Extraction': predicted_extraction,
#                 'Available Groundwater': float(request.form.get('Annual Extractable Ground Water Resource', 0)),
#                 'Irrigation Use': float(request.form.get('Current Annual Ground Water Extraction For Irrigation', 0)),
#                 'Domestic Use': float(request.form.get('Current Annual Ground Water Extraction For Domestic & Industrial Use', 0))
#             }

#             # ✅ 5. Plotting
#             x_vals = list(range(1, 8))
#             y_vals = [i * predicted_extraction / 7 for i in x_vals]

#             plt.figure(figsize=(5, 3))
#             plt.plot(x_vals, y_vals, label="Regression Line", color='blue')
#             plt.scatter([4], [predicted_extraction], color='red', label="Predicted Value", zorder=5)
#             plt.title("Linear Regression Prediction")
#             plt.xlabel("Sample Index")
#             plt.ylabel("Groundwater Extraction")
#             plt.legend()
#             plt.tight_layout()

#             # ✅ 6. Save to static folder
#             if not os.path.exists("static"):
#                 os.makedirs("static")
#             plot_path = "static/plot.png"
#             plt.savefig(plot_path)
#             plt.close()

#         except Exception as e:
#             prediction = f"Error: {e}"
#             decision = "❌ Invalid input. Please enter numeric values."

#     return render_template(
#         'index.html',
#         features=features,
#         prediction=prediction,
#         decision=decision,
#         chart_data=chart_data,
#         plot_path='plot.png' if plot_path else None  # Send only filename to template
#     )

# if __name__ == '__main__':
#     app.run(debug=True, port=5051)









from flask import Flask, render_template, request
import numpy as np
import joblib
import os
import matplotlib

matplotlib.use('Agg')  # ✅ BEFORE importing pyplot
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

app = Flask(__name__)

# Load model
model = joblib.load('groundwater_model.pkl')

# Features used in training
features = [
    'Total Annual Ground Water Recharge',
    'Total Natural Discharges',
    'Annual Extractable Ground Water Resource',
    'Current Annual Ground Water Extraction For Irrigation',
    'Current Annual Ground Water Extraction For Domestic & Industrial Use',
    'Net Ground Water Availability for future use'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    decision = None
    chart_data = None
    plot_path = None

    if request.method == 'POST':
        try:
            # ✅ 1. Input processing
            input_data = [float(request.form[feature]) for feature in features]
            X_input = np.array(input_data).reshape(1, -1)

            # ✅ 2. Model prediction
            predicted_extraction = model.predict(X_input)[0]
            predicted_extraction = max(predicted_extraction, 0)  # No negative values

            # ✅ 3. Decision logic
            if predicted_extraction < 500:
                decision = "✅ Borewell CAN be installed at this location."
            elif predicted_extraction < 1000:
                decision = "⚠️ Borewell MAY be installed — needs local approval."
            else:
                decision = "🚫 Borewell CANNOT be installed — groundwater stress too high."

            prediction = f"{predicted_extraction:.2f}"

            # ✅ 4. Chart data
            chart_data = {
                'Predicted Extraction': predicted_extraction,
                'Available Groundwater': float(request.form.get('Annual Extractable Ground Water Resource', 0)),
                'Irrigation Use': float(request.form.get('Current Annual Ground Water Extraction For Irrigation', 0)),
                'Domestic Use': float(request.form.get('Current Annual Ground Water Extraction For Domestic & Industrial Use', 0))
            }

            # # ✅ 5. Plotting – Based on user input
            # x_user = float(request.form.get('Net Ground Water Availability for future use', 0))

            # x_range = np.linspace(x_user - 200, x_user + 200, 100).reshape(-1, 1)

            # X_plot = np.tile(X_input, (100, 1))
            # X_plot[:, 5] = x_range.flatten()

            # y_range = model.predict(X_plot)

            # plt.figure(figsize=(6, 4))
            # plt.plot(x_range, y_range, label='Regression Curve', color='blue')
            # plt.scatter([x_user], [predicted_extraction], color='red', label='User Input', zorder=5)
            # plt.title("Predicted Groundwater Extraction vs Future Availability")
            # plt.xlabel("Net Ground Water Availability for Future Use")
            # plt.ylabel("Predicted Groundwater Extraction")
            # plt.legend()
            # plt.grid(True)
            # plt.tight_layout()

            # ✅ 6. Save to static folder
            if not os.path.exists("static"):
                os.makedirs("static")
            plot_path = "static/plot.png"
            plt.savefig(plot_path)
            plt.close()

        except Exception as e:
            prediction = f"Error: {e}"
            decision = "❌ Invalid input. Please enter numeric values."

    return render_template(
        'index.html',
        features=features,
        prediction=prediction,
        decision=decision,
        chart_data=chart_data,
        plot_path='plot.png' if plot_path else None  # Send only filename to template
    )

if __name__ == '__main__':
    app.run(debug=True, port=5052)




