import os
import json
import re
import uuid
import base64
import shutil
import urllib.parse
import asyncio
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import (
    FastAPI, HTTPException, Depends, File, UploadFile, Form, Request, status, Cookie
)
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, EmailStr
from jose import JWTError, jwt
from PIL import Image
from io import BytesIO
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="PocketSmart: AI Budget Planner")

SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None
MODEL_NAME = "gemini-2.5-flash"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
os.makedirs("static/uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-Memory Databases for Demonstration
users_db: Dict[str, Dict[str, Any]] = {}
active_sessions: Dict[str, Any] = {}
blacklisted_tokens = set()
user_recommendations: Dict[str, List[Any]] = {}

# Pydantic Models
class RegisterUser(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class UserInDB(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    hashed_password: str

class UserSession(BaseModel):
    username: str
    login_time: datetime
    last_activity: datetime
    token: str
    user_data: Dict[str, Any] = {}

class HomeBudgetInput(BaseModel):
    total_budget: float
    num_lights: int
    num_fans: int
    num_furniture: int
    num_dining_tables: int
    has_living_room: bool = True
    has_kitchen: bool = True
    has_bedroom: bool = True
    additional_requirements: Optional[str] = None

class PartyBudgetInput(BaseModel):
    total_budget: float
    num_guests: int
    party_type: str
    venue_type: Optional[str] = "Home"
    needs_catering: bool = True
    needs_decoration: bool = True
    needs_entertainment: bool = True
    additional_requirements: Optional[str] = None

class JewelryBudgetInput(BaseModel):
    total_budget: float
    occasion: str
    preferences: Optional[str] = None

class RecommendationHistoryItem(BaseModel):
    id: str
    timestamp: str
    recommendation_type: str
    input_summary: Dict[str, Any]
    result_summary: Dict[str, Any]
    full_result: Dict[str, Any]

# Helper Auth Functions
def verify_password(plain_password, hashed_password):
    try:
        scheme, iterations, salt, expected = hashed_password.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac("sha256", plain_password.encode(), bytes.fromhex(salt), int(iterations))
        return hmac.compare_digest(actual.hex(), expected)
    except (AttributeError, ValueError):
        return False

def get_password_hash(password):
    salt = secrets.token_bytes(16)
    iterations = 600_000
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"

def authenticate_user(db, username, password):
    if username not in db:
        return None
    user = db[username]
    if not verify_password(password, user["hashed_password"]):
        return None
    return UserInDB(**user)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_token(request: Request) -> Optional[str]:
    token = request.cookies.get("access_token")
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
    return token

async def get_current_user(request: Request, token: Optional[str] = Depends(get_token)) -> UserInDB:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token or token in blacklisted_tokens:
        raise credentials_exception
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    if username not in users_db:
        raise credentials_exception
    user_data = users_db[username]
    return UserInDB(**user_data)

async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
    return current_user

def extract_json_from_response(text: str) -> dict:
    try:
        match = re.search(r'```(?:json)?\s*({.*?})\s*```', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        match_brace = re.search(r'({.*})', text, re.DOTALL)
        if match_brace:
            return json.loads(match_brace.group(1))
        return json.loads(text)
    except Exception:
        return {"error": "Failed to parse AI response", "raw_text": text}

def save_upload_file(upload_file: UploadFile) -> str:
    ext = os.path.splitext(upload_file.filename)[1]
    filename = f"{uuid.uuid4()}{ext}"
    path = os.path.join("static/uploads", filename)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return path

def save_to_history(username: str, recommendation_type: str, input_data: dict, result: dict):
    if username not in user_recommendations:
        user_recommendations[username] = []
    
    item = RecommendationHistoryItem(
        id=str(uuid.uuid4()),
        timestamp=datetime.utcnow().strftime("%B %d, %Y - %I:%M %p"),
        recommendation_type=recommendation_type,
        input_summary=input_data,
        result_summary={
            "total_budget": result.get("total_budget", 0),
            "remaining_budget": result.get("remaining_budget", 0)
        },
        full_result=result
    )
    user_recommendations[username].append(item)

# AI Recommendation Functions
def generate_plan_or_fallback(prompt: Any, result_key: str, fallback: Any) -> dict:
    if client is None:
        return fallback()
    try:
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        result = extract_json_from_response(response.text)
        if isinstance(result, dict) and isinstance(result.get(result_key), list):
            return result
    except Exception:
        pass
    return fallback()

def fallback_budget_breakdown(total_budget: float, categories: list) -> dict:
    active_categories = [category for category in categories if category[2] > 0]
    weight_total = sum(category[3] for category in active_categories)
    planned_total = total_budget * 0.9
    breakdown = []
    calculation_table = []

    for category, item_name, quantity, weight, description, search_terms in active_categories:
        allocation = planned_total * weight / weight_total
        unit_price = round(allocation / quantity, 2)
        breakdown.append({
            "category": category,
            "allocation": round(allocation, 2),
            "items": [{
                "name": item_name,
                "description": description,
                "estimated_price": unit_price,
                "quantity": quantity,
                "search_terms": search_terms
            }]
        })
        calculation_table.append({
            "category": category,
            "items_count": quantity,
            "total_cost": round(allocation, 2),
            "percentage_of_budget": round(allocation / total_budget * 100, 1)
        })

    return {
        "total_budget": total_budget,
        "remaining_budget": round(total_budget - planned_total, 2),
        "budget_breakdown": breakdown,
        "calculation_table": calculation_table,
        "additional_suggestions": ["Keep the unallocated 10% as a contingency for price changes and delivery costs."],
        "plan_source": "local estimate"
    }

def get_home_recommendations(budget_input: HomeBudgetInput) -> dict:
    prompt = f"""
    I need interior design product recommendations for a home in India with a total budget of ₹{budget_input.total_budget:.2f}.
    Requirements:
    - {budget_input.num_lights} lights/lighting fixtures
    - {budget_input.num_fans} ceiling fans
    - {budget_input.num_furniture} furniture pieces
    - {budget_input.num_dining_tables} dining tables
    Rooms: Living Room: {budget_input.has_living_room}, Kitchen: {budget_input.has_kitchen}, Bedroom: {budget_input.has_bedroom}
    Additional requirements: {budget_input.additional_requirements or "None"}
    Provide detailed breakdown with estimated INR prices and specific search terms.
    Format your response STRICTLY as valid JSON with structure:
    {{
      "total_budget": {budget_input.total_budget},
      "remaining_budget": 0.0,
      "budget_breakdown": [
        {{
          "category": "lighting",
          "allocation": 0.0,
          "items": [
            {{"name": "...", "description": "...", "estimated_price": 0.0, "quantity": 1, "search_terms": "..."}}
          ]
        }}
      ],
      "calculation_table": [
        {{"category": "lighting", "items_count": 1, "total_cost": 0.0, "percentage_of_budget": 0.0}}
      ],
      "additional_suggestions": ["tip 1", "tip 2"]
    }}
    """
    fallback = lambda: fallback_budget_breakdown(budget_input.total_budget, [
        ("Lighting", "Energy-efficient LED light fixtures", budget_input.num_lights, 18, "LED fixtures sized to the room; compare brightness, warranty, and installation costs.", "energy efficient LED ceiling light India"),
        ("Cooling", "Energy-efficient ceiling fans", budget_input.num_fans, 17, "Compare energy ratings, sweep size, and warranty before purchase.", "BEE rated ceiling fan India"),
        ("Furniture", "Everyday home furniture", budget_input.num_furniture, 40, "Prioritize durable, multi-purpose pieces and confirm delivery costs.", "durable home furniture India"),
        ("Dining", "Dining table and seating", budget_input.num_dining_tables, 25, "Choose a size that fits the room and includes the required seating.", "compact dining table set India")
    ])
    result = generate_plan_or_fallback(prompt, "budget_breakdown", fallback)
    
    for category in result.get("budget_breakdown", []):
        for item in category.get("items", []):
            st = item.get("search_terms", "")
            q = urllib.parse.quote_plus(st)
            item["shopping_links"] = {
                "amazon": f"https://www.amazon.in/s?k={q}",
                "flipkart": f"https://www.flipkart.com/search?q={q}",
                "ikea": f"https://www.ikea.com/in/en/search/?q={q}",
                "myntra": f"https://www.myntra.com/search?q={q}",
                "ajio": f"https://www.ajio.com/search/?text={q}"
            }
    return result

def get_party_recommendations(budget_input: PartyBudgetInput) -> dict:
    prompt = f"""
    I need party planning recommendations for India with a total budget of ₹{budget_input.total_budget:.2f}.
    Party Type: {budget_input.party_type}, Guests: {budget_input.num_guests}, Venue: {budget_input.venue_type}
    Catering: {budget_input.needs_catering}, Decoration: {budget_input.needs_decoration}, Entertainment: {budget_input.needs_entertainment}
    Additional: {budget_input.additional_requirements or "None"}
    Format response STRICTLY as valid JSON with structure:
    {{
      "total_budget": {budget_input.total_budget},
      "remaining_budget": 0.0,
      "budget_breakdown": [
        {{
          "category": "venue",
          "allocation": 0.0,
          "items": [{{"name": "...", "description": "...", "estimated_price": 0.0, "quantity": 1, "search_terms": "..."}}]
        }}
      ],
      "venue_suggestions": [
        {{"name": "...", "type": "...", "capacity": {budget_input.num_guests}, "estimated_cost": 0.0, "search_terms": "..."}}
      ],
      "additional_suggestions": ["tip 1"]
    }}
    """
    fallback = lambda: fallback_budget_breakdown(budget_input.total_budget, [
        ("Venue", f"{budget_input.venue_type or 'Home'} venue", 1, 25, "Confirm the booking duration, taxes, and any deposit before paying.", f"{budget_input.venue_type or 'home'} event venue India"),
        ("Catering", f"Food and refreshments for {budget_input.num_guests} guests", budget_input.num_guests if budget_input.needs_catering else 0, 45, "Set a per-person menu limit and confirm dietary requirements.", f"party catering for {budget_input.num_guests} guests India"),
        ("Decoration", f"{budget_input.party_type} decorations", 1 if budget_input.needs_decoration else 0, 20, "Reuse or rent decor where possible to reduce one-time costs.", f"{budget_input.party_type} party decorations India"),
        ("Entertainment", "Party entertainment", 1 if budget_input.needs_entertainment else 0, 10, "Compare a short set or curated playlist against a full event package.", f"{budget_input.party_type} party entertainment India")
    ])
    return generate_plan_or_fallback(prompt, "budget_breakdown", fallback)

def get_jewelry_recommendations(budget_input: JewelryBudgetInput, image_path: Optional[str] = None) -> dict:
    base_prompt = f"""
    I need jewelry recommendations for India with a total budget of ₹{budget_input.total_budget:.2f}.
    Occasion: {budget_input.occasion}. Preferences: {budget_input.preferences or "Not specified"}.
    Provide India-relevant styles, availability, and prices in INR.
    """
    
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path)
        prompt = [base_prompt + "\nAn image of the outfit is uploaded. Suggest jewelry complementing it.", img]
    else:
        prompt = base_prompt + "\nFormat output as JSON with total_budget, jewelry_recommendations (item_type, description, style, estimated_price, search_terms), remaining_budget, and styling_tips."
        
    fallback = lambda: {
        "total_budget": budget_input.total_budget,
        "remaining_budget": round(budget_input.total_budget * 0.1, 2),
        "jewelry_recommendations": [
            {"item_type": "Earrings", "description": f"A versatile pair suited to {budget_input.occasion}; compare metal finish and return policy.", "style": budget_input.preferences or "Classic", "estimated_price": round(budget_input.total_budget * 0.36, 2), "search_terms": f"{budget_input.occasion} earrings India"},
            {"item_type": "Necklace", "description": "A lightweight necklace selected to complement the occasion and outfit.", "style": budget_input.preferences or "Classic", "estimated_price": round(budget_input.total_budget * 0.36, 2), "search_terms": f"{budget_input.occasion} necklace India"},
            {"item_type": "Bracelet", "description": "A simple coordinating accessory within the remaining planned amount.", "style": budget_input.preferences or "Classic", "estimated_price": round(budget_input.total_budget * 0.18, 2), "search_terms": f"{budget_input.occasion} bracelet India"}
        ],
        "styling_tips": ["Keep 10% of the budget aside for taxes, alterations, or delivery."],
        "plan_source": "local estimate"
    }
    result = generate_plan_or_fallback(prompt, "jewelry_recommendations", fallback)
    
    for item in result.get("jewelry_recommendations", []):
        st = item.get("search_terms", "")
        q = urllib.parse.quote_plus(st)
        item["shopping_links"] = {
            "amazon": f"https://www.amazon.in/s?k={q}",
            "flipkart": f"https://www.flipkart.com/search?q={q}",
            "tanishq": f"https://www.tanishq.co.in/search?q={q}",
            "caratlane": f"https://www.caratlane.com/search?q={q}",
            "meesho": f"https://www.meesho.com/search?q={q}"
        }
    return result

# Web & API Routes
@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    try:
        token = await get_token(request)
        if token:
            user = await get_current_user(request, token)
            if user:
                return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    except Exception:
        pass
    return templates.TemplateResponse(request=request, name="login.html", context={"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    try:
        token = await get_token(request)
        if token:
            user = await get_current_user(request, token)
            if user:
                return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    except Exception:
        pass
    error = request.query_params.get("error")
    return templates.TemplateResponse(request=request, name="register.html", context={"request": request, "error": error})

@app.post("/register")
async def register_user_endpoint(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    full_name: Optional[str] = Form(None),
    password: str = Form(...)
):
    if username in users_db:
        return RedirectResponse(url="/register?error=username", status_code=status.HTTP_303_SEE_OTHER)
    hashed_password = get_password_hash(password)
    users_db[username] = {
        "username": username,
        "email": email,
        "full_name": full_name,
        "hashed_password": hashed_password
    }
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    
    active_sessions[user.username] = UserSession(
        username=user.username,
        login_time=datetime.utcnow(),
        last_activity=datetime.utcnow(),
        token=access_token
    )
    
    response = JSONResponse(content={"access_token": access_token, "token_type": "bearer"})
    response.set_cookie(key="access_token", value=access_token, httponly=True, max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60, samesite="lax")
    return response

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    history = user_recommendations.get(current_user.username, [])
    return templates.TemplateResponse(request=request, name="dashboard.html", context={"request": request, "user": current_user, "history": history[:5]})

@app.get("/home-planner", response_class=HTMLResponse)
async def home_planner_page(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    return templates.TemplateResponse(request=request, name="home_planner.html", context={"request": request, "user": current_user})

@app.get("/party-planner", response_class=HTMLResponse)
async def party_planner_page(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    return templates.TemplateResponse(request=request, name="party_planner.html", context={"request": request, "user": current_user})

@app.get("/jewelry-planner", response_class=HTMLResponse)
async def jewelry_planner_page(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    return templates.TemplateResponse(request=request, name="jewelry_planner.html", context={"request": request, "user": current_user})

@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    history = user_recommendations.get(current_user.username, [])
    return templates.TemplateResponse(request=request, name="history.html", context={"request": request, "user": current_user, "history": history})

@app.post("/home-budget")
async def plan_home_budget(budget_input: HomeBudgetInput, current_user: UserInDB = Depends(get_current_active_user)):
    result = get_home_recommendations(budget_input)
    save_to_history(current_user.username, "home", budget_input.dict(), result)
    return result

@app.post("/party-budget")
async def plan_party_budget(budget_input: PartyBudgetInput, current_user: UserInDB = Depends(get_current_active_user)):
    result = get_party_recommendations(budget_input)
    save_to_history(current_user.username, "party", budget_input.dict(), result)
    return result

@app.post("/jewelry-budget")
async def plan_jewelry_budget(
    total_budget: float = Form(...),
    occasion: str = Form(...),
    preferences: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    current_user: UserInDB = Depends(get_current_active_user)
):
    budget_input = JewelryBudgetInput(total_budget=total_budget, occasion=occasion, preferences=preferences)
    image_path = save_upload_file(image) if image else None
    
    result = get_jewelry_recommendations(budget_input, image_path)
    input_data = budget_input.dict()
    if image:
        input_data["image"] = image.filename
    save_to_history(current_user.username, "jewelry", input_data, result)
    return result

@app.post("/logout")
async def logout(request: Request):
    token = await get_token(request)
    if token:
        blacklisted_tokens.add(token)
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if username in active_sessions:
                del active_sessions[username]
        except JWTError:
            pass
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie(key="access_token")
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)