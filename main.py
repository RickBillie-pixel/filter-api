"""
Filter API - Removes irrelevant elements from extracted data
Implements knowledge base rules for filtering and data cleaning
Filters out short lines, tiny symbols, decorative noise, etc.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("filter_api")

# Knowledge Base - Filtering Constants
MIN_WALL_LENGTH = 0.1  # meters - minimum wall length to consider
MIN_WALL_THICKNESS = 0.005  # meters - minimum wall thickness to consider
MIN_ROOM_AREA = 0.1  # m² - minimum room area to consider
MIN_COMPONENT_CONFIDENCE = 0.3  # minimum confidence for components
MIN_SYMBOL_CONFIDENCE = 0.3  # minimum confidence for installation symbols
MIN_TEXT_SIZE = 6  # points - minimum text size to consider
MAX_TEXT_LENGTH = 50  # characters - maximum text length for labels

app = FastAPI(
    title="Data Filtering API",
    description="Removes irrelevant elements from extracted vector data",
    version="1.0.0",
)

class Wall(BaseModel):
    type: str
    label_code: Optional[str] = None
    label_nl: Optional[str] = None
    label_en: Optional[str] = None
    label_type: Optional[str] = None
    thickness_meters: Optional[float] = None
    properties: Optional[Dict[str, Any]] = None
    classification: Optional[Dict[str, Any]] = None
    orientation: Optional[str] = None
    wall_type: Optional[str] = None
    confidence: Optional[float] = None
    reason: Optional[str] = None

class Room(BaseModel):
    name: str
    room_type: Optional[str] = None
    room_code: Optional[str] = None
    area_m2: float
    polygon: List[Dict[str, float]]
    confidence: float
    reason: str
    has_access: Optional[bool] = None
    label_code: Optional[str] = None
    label_type: Optional[str] = None
    label_nl: Optional[str] = None
    label_en: Optional[str] = None

class Component(BaseModel):
    type: str
    label_code: Optional[str] = None
    label_type: Optional[str] = None
    label_nl: Optional[str] = None
    label_en: Optional[str] = None
    position: Optional[Dict[str, float]] = None
    width_m: Optional[float] = None
    height_m: Optional[float] = None
    confidence: float
    reason: str
    properties: Optional[Dict[str, Any]] = None

class Symbol(BaseModel):
    type: str
    label_code: Optional[str] = None
    label_type: Optional[str] = None
    label_nl: Optional[str] = None
    label_en: Optional[str] = None
    position: Optional[Dict[str, float]] = None
    text: Optional[str] = None
    bbox: Optional[Dict[str, float]] = None
    confidence: float
    reason: str
    source: Optional[str] = None
    shape: Optional[str] = None

class TextItem(BaseModel):
    text: str
    position: Dict[str, float]
    font_size: float
    font_name: str
    color: List[float] = [0, 0, 0]
    bbox: Dict[str, float]

class DrawingItem(BaseModel):
    type: str
    p1: Optional[Dict[str, float]] = None
    p2: Optional[Dict[str, float]] = None
    p3: Optional[Dict[str, float]] = None
    rect: Optional[Dict[str, float]] = None
    length: Optional[float] = None
    color: List[float] = [0, 0, 0]
    width: Optional[float] = 1.0
    area: Optional[float] = None
    fill: List[Any] = []

class Drawings(BaseModel):
    lines: List[DrawingItem]
    rectangles: List[DrawingItem]
    curves: List[DrawingItem]

class PageData(BaseModel):
    page_number: int
    page_size: Dict[str, float]
    drawings: Drawings
    texts: List[TextItem]
    is_vector: bool = True
    processing_time_ms: Optional[int] = None

class FilterRequest(BaseModel):
    pages: List[PageData]
    walls: List[List[Wall]]
    rooms: List[List[Room]]
    components: List[List[Component]]
    symbols: List[List[Symbol]]
    scale_m_per_pixel: float = 1.0

class FilterResponse(BaseModel):
    pages: List[Dict[str, Any]]

@app.post("/filter-data/", response_model=FilterResponse)
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
            
            # Get data for current page
            walls = request.walls[i] if i < len(request.walls) else []
            rooms = request.rooms[i] if i < len(request.rooms) else []
            components = request.components[i] if i < len(request.components) else []
            symbols = request.symbols[i] if i < len(request.symbols) else []
            
            filtered = _filter_irrelevant_elements(
                page_data, walls, rooms, components, 
                symbols, request.scale_m_per_pixel
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

def _filter_irrelevant_elements(page_data: PageData, walls: List[Wall], 
                              rooms: List[Room], components: List[Component], 
                              symbols: List[Symbol], scale: float) -> Dict[str, Any]:
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
    
    # Step 1: Filter walls
    logger.info(f"Filtering walls: starting with {len(walls)} walls")
    filtered_walls = []
    for wall in walls:
        # Skip walls with unknown type
        if wall.type == "unknown":
            continue
        
        # Check wall properties
        if wall.properties:
            # Remove very short walls
            if wall.properties.get("length_meters", 0) < MIN_WALL_LENGTH:
                logger.debug(f"Filtering out wall: too short ({wall.properties.get('length_meters', 0)}m)")
                continue
            
            # Remove very thin walls
            if wall.thickness_meters and wall.thickness_meters < MIN_WALL_THICKNESS:
                logger.debug(f"Filtering out wall: too thin ({wall.thickness_meters}m)")
                continue
        
        filtered_walls.append(wall.dict() if hasattr(wall, 'dict') else wall)
    
    logger.info(f"Filtered walls: {len(filtered_walls)} walls remaining")
    
    # Step 2: Filter rooms
    logger.info(f"Filtering rooms: starting with {len(rooms)} rooms")
    filtered_rooms = []
    for room in rooms:
        # Skip rooms with unknown type
        if room.name == "unknown" and room.room_type == "unknown":
            continue
        
        # Remove very small rooms
        if room.area_m2 < MIN_ROOM_AREA:
            logger.debug(f"Filtering out room: too small ({room.area_m2}m²)")
            continue
        
        filtered_rooms.append(room.dict() if hasattr(room, 'dict') else room)
    
    logger.info(f"Filtered rooms: {len(filtered_rooms)} rooms remaining")
    
    # Step 3: Filter components
    logger.info(f"Filtering components: starting with {len(components)} components")
    filtered_components = []
    for component in components:
        # Skip components with unknown type
        if component.type == "unknown":
            continue
        
        # Remove components with very low confidence
        if component.confidence < MIN_COMPONENT_CONFIDENCE:
            logger.debug(f"Filtering out component: low confidence ({component.confidence})")
            continue
        
        filtered_components.append(component.dict() if hasattr(component, 'dict') else component)
    
    logger.info(f"Filtered components: {len(filtered_components)} components remaining")
    
    # Step 4: Filter symbols
    logger.info(f"Filtering symbols: starting with {len(symbols)} symbols")
    filtered_symbols = []
    for symbol in symbols:
        # Skip symbols with unknown type
        if symbol.type == "unknown":
            continue
        
        # Remove symbols with very low confidence
        if symbol.confidence < MIN_SYMBOL_CONFIDENCE:
            logger.debug(f"Filtering out symbol: low confidence ({symbol.confidence})")
            continue
        
        filtered_symbols.append(symbol.dict() if hasattr(symbol, 'dict') else symbol)
    
    logger.info(f"Filtered symbols: {len(filtered_symbols)} symbols remaining")
    
    # Step 5: Filter texts (remove decorative or irrelevant text)
    logger.info(f"Filtering texts: starting with {len(page_data.texts)} texts")
    unlinked_texts = []
    for text in page_data.texts:
        text_content = text.text.strip()
        
        # Skip empty text
        if not text_content:
            continue
        
        # Skip very small text (likely decorative)
        if text.font_size < MIN_TEXT_SIZE:
            logger.debug(f"Filtering out text: too small ({text.font_size}pt)")
            continue
        
        # Skip text that's too long (likely not a label)
        if len(text_content) > MAX_TEXT_LENGTH:
            logger.debug(f"Filtering out text: too long ({len(text_content)} chars)")
            continue
        
        # Skip text that's all numbers (likely dimensions)
        if text_content.replace(".", "").replace(",", "").replace(" ", "").isdigit():
            logger.debug(f"Filtering out text: numeric only")
            continue
        
        unlinked_texts.append(text.dict() if hasattr(text, 'dict') else text)
    
    logger.info(f"Filtered texts: {len(unlinked_texts)} texts remaining")
    
    # Step 6: Validate data consistency
    validation_errors = _validate_data_consistency(
        filtered_walls, filtered_rooms, filtered_components, filtered_symbols
    )
    errors.extend(validation_errors)
    
    # Final consistency check for symbols and components that may be duplicates
    filtered_symbols = _remove_duplicate_symbols(filtered_symbols)
    filtered_components = _remove_duplicate_components(filtered_components)
    
    logger.info(f"Final filtered counts: {len(filtered_walls)} walls, {len(filtered_rooms)} rooms, "
               f"{len(filtered_components)} components, {len(filtered_symbols)} symbols, "
               f"{len(unlinked_texts)} texts, {len(errors)} errors")
    
    return {
        "walls": filtered_walls,
        "rooms": filtered_rooms,
        "components": filtered_components,
        "symbols": filtered_symbols,
        "unlinked_texts": unlinked_texts,
        "errors": errors
    }

def _validate_data_consistency(walls: List[Dict[str, Any]], rooms: List[Dict[str, Any]], 
                              components: List[Dict[str, Any]], symbols: List[Dict[str, Any]]) -> List[str]:
    """
    Validate consistency between different data types
    
    Args:
        walls: Filtered walls
        rooms: Filtered rooms
        components: Filtered components
        symbols: Filtered symbols
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    # Check 1: Components should be associated with walls
    for component in components:
        if component.get("type") in ["door", "window", "sliding_door"] and not component.get("wall_reference"):
            errors.append(f"Component {component.get('type')} has no wall reference")
    
    # Check 2: Rooms should have reasonable sizes and shapes
    for room in rooms:
        # Check if room polygon has at least 3 points
        if len(room.get("polygon", [])) < 3:
            errors.append(f"Room {room.get('name')} has invalid polygon (less than 3 points)")
        
        # Check for unreasonably large rooms (likely errors)
        if room.get("area_m2", 0) > 1000:
            errors.append(f"Room {room.get('name')} has suspiciously large area ({room.get('area_m2')}m²)")
    
    # Check 3: Walls should have consistent types
    exterior_count = sum(1 for w in walls if w.get("wall_type") == "exterior")
    if exterior_count == 0 and len(walls) > 5:
        errors.append("No exterior walls detected in a drawing with multiple walls")
    
    return errors

def _remove_duplicate_symbols(symbols: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate installation symbols based on position
    
    Args:
        symbols: List of installation symbols
        
    Returns:
        List with duplicates removed
    """
    if not symbols:
        return []
        
    # Group symbols by position (with tolerance)
    position_groups = {}
    tolerance = 10  # pixels
    
    for symbol in symbols:
        if not symbol.get("position"):
            continue
            
        x, y = symbol.get("position", {}).get("x", 0), symbol.get("position", {}).get("y", 0)
        found_group = False
        
        for group_pos, group in position_groups.items():
            gx, gy = group_pos
            if abs(x - gx) < tolerance and abs(y - gy) < tolerance:
                group.append(symbol)
                found_group = True
                break
        
        if not found_group:
            position_groups[(x, y)] = [symbol]
    
    # For each group, keep only the highest confidence symbol
    unique_symbols = []
    for group in position_groups.values():
        if len(group) > 1:
            # Keep symbol with highest confidence
            best_symbol = max(group, key=lambda s: s.get("confidence", 0))
            unique_symbols.append(best_symbol)
        else:
            unique_symbols.extend(group)
    
    return unique_symbols

def _remove_duplicate_components(components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate components based on position
    
    Args:
        components: List of components
        
    Returns:
        List with duplicates removed
    """
    if not components:
        return []
        
    # Group components by position (with tolerance)
    position_groups = {}
    tolerance = 10  # pixels
    
    for component in components:
        if not component.get("position"):
            continue
            
        x, y = component.get("position", {}).get("x", 0), component.get("position", {}).get("y", 0)
        found_group = False
        
        for group_pos, group in position_groups.items():
            gx, gy = group_pos
            if abs(x - gx) < tolerance and abs(y - gy) < tolerance:
                group.append(component)
                found_group = True
                break
        
        if not found_group:
            position_groups[(x, y)] = [component]
    
    # For each group, keep only the highest confidence component
    unique_components = []
    for group in position_groups.values():
        if len(group) > 1:
            # Keep component with highest confidence
            best_component = max(group, key=lambda c: c.get("confidence", 0))
            unique_components.append(best_component)
        else:
            unique_components.extend(group)
    
    return unique_components

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Data Filtering API",
        "version": "1.0.0",
        "endpoints": {
            "/filter-data/": "Filter irrelevant elements from extracted data",
            "/health/": "Health check"
        }
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "filter-api",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)