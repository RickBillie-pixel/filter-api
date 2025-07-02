"""
Filter API - Removes irrelevant elements from extracted data
Filters out short lines, tiny symbols, decorative noise, etc.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("filter_api")

app = FastAPI(
    title="Data Filtering API",
    description="Removes irrelevant elements from extracted vector data",
    version="1.0.0",
)

class PageData(BaseModel):
    page_number: int
    drawings: List[Dict[str, Any]]
    texts: List[Dict[str, Any]]

class Wall(BaseModel):
    p1: Dict[str, float]
    p2: Dict[str, float]
    wall_thickness: float
    wall_length: float
    wall_type: str
    confidence: float
    reason: str

class Room(BaseModel):
    name: str
    area_m2: float
    polygon: List[Dict[str, float]]
    confidence: float
    reason: str

class Component(BaseModel):
    type: str
    position: Dict[str, float]
    confidence: float
    reason: str

class Symbol(BaseModel):
    type: str
    position: Dict[str, float]
    confidence: float
    reason: str

class FilterRequest(BaseModel):
    pages: List[PageData]
    walls: List[List[Wall]]
    rooms: List[List[Room]]
    components: List[List[Component]]
    symbols: List[List[Symbol]]
    scale_m_per_pixel: float = 1.0

@app.post("/filter-data/")
async def filter_data(request: FilterRequest):
    """
    Filter out irrelevant elements from extracted data
    
    Args:
        request: JSON with all extracted data and scale information
        
    Returns:
        JSON with filtered data for each page
    """
    try:
        logger.info(f"Filtering data for {len(request.pages)} pages with scale {request.scale_m_per_pixel}")
        
        results = []
        
        for i, page_data in enumerate(request.pages):
            logger.info(f"Filtering data on page {page_data.page_number}")
            
            # Convert to dictionaries for processing
            walls_dict = [wall.dict() for wall in request.walls[i]]
            rooms_dict = [room.dict() for room in request.rooms[i]]
            components_dict = [comp.dict() for comp in request.components[i]]
            symbols_dict = [sym.dict() for sym in request.symbols[i]]
            
            filtered = _filter_irrelevant_elements(
                page_data, walls_dict, rooms_dict, components_dict, 
                symbols_dict, request.scale_m_per_pixel
            )
            
            results.append({
                "page_number": page_data.page_number,
                "walls": filtered["walls"],
                "rooms": filtered["rooms"],
                "components": filtered["components"],
                "symbols": filtered["symbols"],
                "unlinked_texts": filtered["unlinked_texts"],
                "errors": filtered["errors"]
            })
        
        logger.info(f"Successfully filtered data for {len(results)} pages")
        return {"pages": results}
        
    except Exception as e:
        logger.error(f"Error filtering data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def _filter_irrelevant_elements(page_data: PageData, walls: List[Dict[str, Any]], 
                              rooms: List[Dict[str, Any]], components: List[Dict[str, Any]], 
                              symbols: List[Dict[str, Any]], scale: float) -> Dict[str, Any]:
    """
    Filter out irrelevant elements using rule-based approach
    
    Args:
        page_data: Page data containing drawings and texts
        walls: List of wall dictionaries
        rooms: List of room dictionaries
        components: List of component dictionaries
        symbols: List of symbol dictionaries
        scale: Scale factor in meters per pixel
        
    Returns:
        Dictionary with filtered data
    """
    errors = []
    
    # Filter walls
    filtered_walls = []
    for wall in walls:
        if wall.get("type") == "unknown":
            continue
        
        # Remove very short walls
        if wall.get("wall_length", 0) < 0.1:  # Less than 10cm
            continue
        
        # Remove very thin walls
        if wall.get("wall_thickness", 0) < 0.005:  # Less than 5mm
            continue
        
        filtered_walls.append(wall)
    
    # Filter rooms
    filtered_rooms = []
    for room in rooms:
        if room.get("type") == "unknown":
            continue
        
        # Remove very small rooms
        if room.get("area_m2", 0) < 0.1:  # Less than 0.1 m²
            continue
        
        filtered_rooms.append(room)
    
    # Filter components
    filtered_components = []
    for component in components:
        if component.get("type") == "unknown":
            continue
        
        # Remove components with very low confidence
        if component.get("confidence", 0) < 0.3:
            continue
        
        filtered_components.append(component)
    
    # Filter symbols
    filtered_symbols = []
    for symbol in symbols:
        if symbol.get("type") == "unknown":
            continue
        
        # Remove symbols with very low confidence
        if symbol.get("confidence", 0) < 0.3:
            continue
        
        filtered_symbols.append(symbol)
    
    # Filter texts (remove decorative or irrelevant text)
    unlinked_texts = []
    for text in page_data.texts:
        text_content = text["text"].strip()
        
        # Skip empty text
        if not text_content:
            continue
        
        # Skip very small text (likely decorative)
        if text.get("size", 0) < 6:
            continue
        
        # Skip text that's too long (likely not a label)
        if len(text_content) > 50:
            continue
        
        # Skip text that's all numbers (likely dimensions)
        if text_content.replace(".", "").replace(",", "").isdigit():
            continue
        
        unlinked_texts.append(text)
    
    logger.info(f"Filtered: {len(filtered_walls)} walls, {len(filtered_rooms)} rooms, "
               f"{len(filtered_components)} components, {len(filtered_symbols)} symbols, "
               f"{len(unlinked_texts)} texts")
    
    return {
        "walls": filtered_walls,
        "rooms": filtered_rooms,
        "components": filtered_components,
        "symbols": filtered_symbols,
        "unlinked_texts": unlinked_texts,
        "errors": errors
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "filter-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006) 