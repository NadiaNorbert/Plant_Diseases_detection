from agno.agent import Agent
from agno.tools.openweather import OpenWeatherTools
from agno.models.ollama import Ollama
import ollama

# Weather API Key
weather_api = "your api_key"

# Setup weather agent
agent = Agent(
    model=Ollama(id="llama3.2"),
    tools=[
        OpenWeatherTools(
            units="standard",
            api_key=weather_api,
            current_weather=True,
            forecast=True,
        )
    ],
    markdown=True,
    description="""
        Consider The Location Given by the User and Give just the Weather Forecast Prediction...
        Don't need Comphrehensive Summary or Description but yet need the Details in a concise Manner.
                                   Instruction to Follow:
                                    - Give Each Prediction for the Current Day as well as for the Next 2 Days.
                                    - First Give the Temperature on the Place and Give Temperature Prediction for the Next Day.
                                    - Give Humidity for Prediction for the Current day as well as for the Next Days.
                                    - Give Precaution Alert Message if there is any Natural Disaster Alert.
                                    - Include information which contains whether forecast details and Give response like:
                                            • Temperature: [include whether forecast prediction or say can't predict at this Moment]                                                                                                               ┃
                                            • Humidity: [include whether forecast prediction or say can't predict at this Moment]                                                                                                                                             ┃
                                            • Wind Speed: [include whether forecast prediction or say can't predict at this Moment]                                                                                                                                     ┃
                                            • Visibility: [include whether forecast prediction or say can't predict at this Moment]                                                                                                                                         ┃
                                            • Weather Condition: [include whether forecast prediction or say can't predict at this Moment]
                                            • Disaster Alert: [include whether forecast prediction or say can't predict at this Moment].
                                     - [**Important**] Give the Output ONLY IN STRING FORMAT, STRICTLY NO JSON FORMAT.
                                     - [**Important**] If You Can't Predict The Whether Forecast Prediction Say ' I Can't Predict the Whether At The Moment'.
                                     - [**Important**] Clearly State The Date of The Days and Organize the Response in the Given Format:
                                           Day: Today
                                           Date: [date of the Current Day]
                                                • Temperature: [include whether forecast prediction or say can't predict at this Moment]                                                                                                               ┃
                                                • Humidity: [include whether forecast prediction or say can't predict at this Moment]                                                                                                                                             ┃
                                                • Wind Speed: [include whether forecast prediction or say can't predict at this Moment]                                                                                                                                     ┃
                                                • Visibility: [include whether forecast prediction or say can't predict at this Moment]                                                                                                                                         ┃
                                                • Weather Condition: [include whether forecast prediction or say can't predict at this Moment]
                                                • Disaster Alert: [include whether forecast prediction or say can't predict at this Moment]
                                                
                                           Day: Next Day
                                           Date: [date of the Next Day]
                                                • Temperature: [include whether forecast prediction or say can't predict at this Moment]                                                                                                               ┃
                                                • Humidity: [include whether forecast prediction or say can't predict at this Moment]                                                                                                                                             ┃
                                                • Wind Speed: [include whether forecast prediction or say can't predict at this Moment]                                                                                                                                     ┃
                                                • Visibility: [include whether forecast prediction or say can't predict at this Moment]                                                                                                                                         ┃
                                                • Weather Condition: [include whether forecast prediction or say can't predict at this Moment]
                                                • Disaster Alert: [include whether forecast prediction or say can't predict at this Moment]
                                 """
)


def get_precaution_weather_report(location: str) -> dict:
    # Get weather forecast
    forecast = str(agent.print_response(location, markdown=True))

    # Prompt for precaution advice
    precaution_prompt = f"""
        whether_forecast_report: "{forecast}"
        Analyse the whether_forecast_report and Give the precaution for the Farmer to Prevent the Crops
        From getting damaged from Heavy Rainfall,High Temperature,High/Low Humidity
        or any other Natural Disaster/Calamities.Finally ask the user 
        'Let me Know if You have Any Doubts Regarding The Precaution Plan or Need Clarification 
         on the Precaution Tips Given'
    """

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "system",
                "content":"""you are a Precaution Giving Ai Assistant, you need to Analyse the whether_forecast_report and 
                              give the Precautions to be Taken by the Farmers to prevent the crops from getting damaged
                              If The Whether Forecast is None Tell the user to Watch Tv or use Mobile to Monitor the whether 
                              Forecast and Contact the Professional Agricultural Officier to Sort out their Queries
                              Finally ask the user 
                              'Let me Know if You have Any Doubts Regarding The Precaution Plan or Need Clarification 
                               on the Precaution Tips Given'"""
            },
            {
                "role": "user",
                "content": precaution_prompt
            }
        ]
    )

    return {
        "forecast": forecast,
        "precaution_advice": response["message"]["content"]
    }
