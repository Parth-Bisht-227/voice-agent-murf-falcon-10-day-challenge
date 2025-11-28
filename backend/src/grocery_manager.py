import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger("grocery_manager")

@dataclass
class CartItem:
    price: float
    quantity: int
    category: str

class GroceryManager:
    """
    Manages a grocery shopping cart.
    """
    def __init__(self):
        # The State: Key = Item Name, Value = CartItem
        self.cart: Dict[str, CartItem] = {}

        current_dir = Path(__file__).parent
        catalog_path = current_dir / "catalog.json"

        # Load the catalog
        try:
            with open(catalog_path,"r") as f:
                self.catalog_data = json.load(f)
                
                # Flatten for easy lookup    
                self.product_lookup = {}
                for category, items in self.catalog_data.items():
                    for item in items:
                        self.product_lookup[item["name"].lower()] = {
                            "name": item["name"], # Official name
                            "price": item["price"],
                            "category": category
                        }

        except FileNotFoundError:
            logger.error("catalog.json not found!")
            self.catalog_data = {}
            self.product_lookup = {}
        
        self.orders_dir = Path("orders")
        self.orders_dir.mkdir(exist_ok = True)
    
    def get_catalog_str(self) -> str:
        return json.dumps(self.catalog_data, indent = 2)
    
    def add_items(self, item_names: List[str], quantities: List[int]) -> dict:
        added_items = []
        failed_items = []

        # Ensuring that the lists are of same length
        if len(quantities) < len(item_names):
            quantities = quantities+ [1] *(len(item_names) - len(quantities))
        
        for name, qty in zip(item_names, quantities):
            clean_name = name.lower().strip()
            
            if clean_name in self.product_lookup:
                product = self.product_lookup[clean_name]
                official_name = product["name"]
                
                if official_name in self.cart:
                    self.cart[official_name].quantity += qty
                else:
                    self.cart[official_name] = CartItem(
                        price=product["price"],
                        quantity=qty,
                        category=product["category"]
                    )
                added_items.append(f"{qty}x {official_name}")
            else:
                failed_items.append(name)

        msg = ""
        if added_items:
            msg += f"Added: {', '.join(added_items)}. "
        if failed_items:
            msg += f"Could not find: {', '.join(failed_items)}."
        
        return {"success": True, "message": msg}

    def remove_item(self, item_name: str) -> dict:
        target = None
        for name in self.cart.keys():
            if item_name.lower() in name.lower():
                target = name
                break
        
        if target:
            del self.cart[target]
            return {"success": True, "message": f"Removed {target} from your cart."}
        else:
            return {"success": False, "message": f"{item_name} wasn't in your cart."}

    def get_cart_summary(self) -> str:
        if not self.cart:
            return "Your cart is currently empty."

        summary = "Current Cart:\n"
        total = 0.0
        
        for name, item in self.cart.items():
            cost = item.price * item.quantity
            total += cost
            summary += f"- {name} (x{item.quantity}): Rs. {cost:.2f}\n"
        
        summary += f"\nTotal: Rs. {total:.2f}"
        return summary

    def checkout(self) -> dict:
        if not self.cart:
            return {"success": False, "message": "Cart is empty."}

        total_price = sum(item.price * item.quantity for item in self.cart.values())

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        order_data = {
            "order_id": f"ORD-{timestamp}",
            "timestamp": datetime.now().isoformat(),
            "items": [
                {
                    "name": name,
                    "quantity": item.quantity,
                    "price_per_unit": item.price,
                    "total": item.price * item.quantity
                }
                for name, item in self.cart.items()
            ],
            "total_price": total_price,
            "currency": "INR",
            "status": "placed"
        }

        filename = f"grocery_order_{timestamp}.json"
        filepath = self.orders_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(order_data, f, indent=2)

        self.cart = {} # Clear cart

        return {
            "success": True, 
            "message": f"Order placed! Total is Rs. {total_price:.2f}. Receipt saved as {filename}."
        }