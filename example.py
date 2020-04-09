from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from starlette.routing import Route
from fastai.vision import (
    open_image,
    load_learner
)
from io import BytesIO
import aiohttp  
from starlette.routing import Mount
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
import uvicorn
import os

learner = load_learner('./')

mapping = {
    "000":"அ",
"001":"ஆ",
"002":"இ",
"003":"ஈ",
"004":"உ",
"005":"ஊ",
"006":"எ",
"007":"ஏ",
"008":"ஐ",
"009":"ஒ",
"010":"ஓ",
"011":"ஃ",
"012":"க",
"013":"ங",
"014":"ச",
"015":"ஞ",
"016":"ட",
"017":"ண",
"018":"த",
"019":"ந",
"020":"ப",
"021":"ம",
"022":"ய",
"023":"ர",
"024":"ல",
"025":"வ",
"026":"ழ",
"027":"ள",
"028":"ற",
"029":"ன",
"030":"ஸ",
"031":"ஷ",
"032":"ஜ",
"033":"ஹ",
"034":"க்ஷ",
"035":"கி",
"036":"ஙி",
"037":"சி",
"038":"ஞி",
"039":"டி",
"040":"ணி",
"041":"தி",
"042":"நி",
"043":"பி",
"044":"மி",
"045":"யி",
"046":"ரி",
"047":"லி",
"048":"வி",
"049":"ழி",
"050":"ளி",
"051":"றி",
"052":"னி",
"053":"ஸி",
"054":"ஷி",
"055":"ஜி",
"056":"ஹி",
"057":"க்ஷி",
"058":"கீ",
"059":"ஙீ",
"060":"சீ",
"061":"ஞீ",
"062":"டீ",
"063":"ணீ",
"064":"தீ",
"065":"நீ",
"066":"பீ",
"067":"மீ",
"068":"யீ",
"069":"ரீ",
"070":"லீ",
"071":"வீ",
"072":"ழீ",
"073":"ளீ",
"074":"றீ",
"075":"னீ",
"076":"ஸீ",
"077":"ஷீ",
"078":"ஜீ",
"079":"ஹீ",
"080":"க்ஷீ",
"081":"கு",
"082":"ஙு",
"083":"சு",
"084":"ஞு",
"085":"டு",
"086":"ணு",
"087":"து",
"088":"நு",
"089":"பு",
"090":"மு",
"091":"யு",
"092":"ரு",
"093":"லு",
"094":"வு",
"095":"ழு",
"096":"ளு",
"097":"று",
"098":"னு",
"099":"கூ",
"100":"ஙூ",
"101":"சூ",
"102":"ஞூ",
"103":"டூ",
"104":"ணூ",
"105":"தூ",
"106":"நூ",
"107":"பூ",
"108":"மூ",
"109":"யூ",
"110":"ரூ",
"111":"லூ",
"112":"வூ",
"113":"ழூ",
"114":"ளூ",
"115":"றூ",
"116":"னூ",
"117":"ா",
"118":"ெ",
"119":"ே",
"120":"ை",
"121":"ஸ்ரீ",
"122":"ஸு",
"123":"ஷு",
"124":"ஜு",
"125":"ஹு",
"126":"க்ஷு",
"127":"ஸூ",
"128":"ஷூ",
"129":"ஜூ",
"130":"ஹூ",
"131":"க்ஷூ",
"132":"க்",
"133":"ங்",
"134":"ச்",
"135":"ஞ்",
"136":"ட்",
"137":"ண்",
"138":"த்",
"139":"ந்",
"140":"ப்",
"141":"ம்",
"142":"ய்",
"143":"ர்",
"144":"ள்",
"145":"வ்",
"146":"ழ்",
"147":"ள்",
"148":"ற்",
"149":"ன்",
"150":"ஸ்",
"151":"ஷ்",
"152":"ஜ்",
"153":"ஹ்",
"154":"க்ஷ",
"155":"ஔ"
}

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    _,_,losses = learner.predict(img)
    predictions = sorted(
            zip(learner.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        )
    
    return JSONResponse({
        "rank1": mapping[predictions[0][0]],
        "rank2": mapping[predictions[1][0]],
        "rank3": mapping[predictions[2][0]],
        "predictions":  predictions
    })


async def homepage(request):
    return HTMLResponse(
        """
        <form action="/upload" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
        Or submit a URL:
        <form action="/classify-url" method="get">
            <input type="url" name="url">
            <input type="submit" value="Fetch and analyze image">
        </form>
    """)    


async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)    


app = Starlette(debug=True, routes=[  
  Route('/upload', upload, methods=['POST']),
  Route('/', RedirectResponse(url='/index.html')),
  Mount('/', app=StaticFiles(directory='build'), name="build    ")
])


app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_headers=["*"], allow_methods=["*"]
)

@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)


port = int(os.environ.get("PORT", 8080))

uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")