import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("order_manager")


@dataclass
class OrderState:
    """
    Dataclass to represent the state of a coffee order.
    
    This is the central data structure that tracks what the customer has ordered.
    Each field starts as None and gets filled as the user provides information.
    """
    drinkType: Optional[str] = None
    size: Optional[str] = None
    milk: Optional[str] = None
    extras: list[str] = field(default_factory=list)
    name: Optional[str] = None


class OrderManager:
    """
    Manages coffee orders, including validation, state tracking, and persistence.
    
    This class provides methods to:
    - Update order fields with validation
    - Check if order is complete
    - Get missing fields
    - Save completed orders to JSON files
    """

    # Valid drink types at this coffee shop
    VALID_DRINKS = [
        "Espresso",
        "Americano",
        "Latte",
        "Cappuccino",
        "Macchiato",
        "Flat White",
        "Mocha",
        "Caramel Latte",
        "Vanilla Latte",
        "Hazelnut Latte",
    ]

    # Valid cup sizes
    VALID_SIZES = ["Small", "Medium", "Large"]

    # Valid milk options
    VALID_MILK = [
        "Whole Milk",
        "Oat Milk",
        "Almond Milk",
        "Skim Milk",
        "No Milk",
        "Soy Milk",
    ]

    # Valid extras/toppings
    VALID_EXTRAS = [
        "Extra Shot",
        "Whipped Cream",
        "Caramel Drizzle",
        "Chocolate Powder",
        "Cinnamon",
        "Vanilla Syrup",
        "Hazelnut Syrup",
        "Foam",
    ]

    def __init__(self):
        """Initialize the order manager with an empty order."""
        self.order = OrderState()
        # Create orders directory if it doesn't exist
        self.orders_dir = Path("orders")
        self.orders_dir.mkdir(exist_ok=True)

    def set_drink_type(self, drink: str) -> dict:
        """
        Set the drink type for the order.
        
        Logic:
        1. Normalize the input (capitalize properly)
        2. Validate against VALID_DRINKS
        3. Update order.drinkType if valid
        4. Return success/error message for LLM feedback
        
        Args:
            drink: The drink type requested by customer
            
        Returns:
            Dictionary with success status and message
        """
        # Try to find a matching drink (case-insensitive)
        matched_drink = None
        for valid_drink in self.VALID_DRINKS:
            if drink.lower() in valid_drink.lower():
                matched_drink = valid_drink
                break

        if not matched_drink:
            error_msg = (
                f"Sorry, we don't have '{drink}'. "
                f"Available drinks: {', '.join(self.VALID_DRINKS[:5])}..."
            )
            logger.warning(f"Invalid drink requested: {drink}")
            return {"success": False, "message": error_msg}

        self.order.drinkType = matched_drink
        logger.info(f"Drink set to: {matched_drink}")
        return {"success": True, "message": f"Great! I'll make you a {matched_drink}."}

    def set_size(self, size: str) -> dict:
        """
        Set the cup size for the order.
        
        Logic:
        1. Normalize input (case-insensitive matching)
        2. Validate against VALID_SIZES
        3. Update order.size if valid
        4. Return feedback message
        
        Args:
            size: The cup size ("Small", "Medium", "Large")
            
        Returns:
            Dictionary with success status and message
        """
        # Case-insensitive matching
        matched_size = None
        for valid_size in self.VALID_SIZES:
            if size.lower() == valid_size.lower():
                matched_size = valid_size
                break

        if not matched_size:
            error_msg = f"Invalid size. Please choose: {', '.join(self.VALID_SIZES)}"
            logger.warning(f"Invalid size requested: {size}")
            return {"success": False, "message": error_msg}

        self.order.size = matched_size
        logger.info(f"Size set to: {matched_size}")
        return {"success": True, "message": f"Perfect! A {matched_size} it is."}

    def set_milk_option(self, milk: str) -> dict:
        """
        Set the milk type for the order.
        
        Logic:
        1. Normalize input (case-insensitive matching)
        2. Validate against VALID_MILK
        3. Update order.milk if valid
        4. Return feedback message
        
        Args:
            milk: The milk option
            
        Returns:
            Dictionary with success status and message
        """
        matched_milk = None
        for valid_milk in self.VALID_MILK:
            if milk.lower() in valid_milk.lower():
                matched_milk = valid_milk
                break

        if not matched_milk:
            error_msg = f"We offer: {', '.join(self.VALID_MILK)}"
            logger.warning(f"Invalid milk option: {milk}")
            return {"success": False, "message": error_msg}

        self.order.milk = matched_milk
        logger.info(f"Milk set to: {matched_milk}")
        return {"success": True, "message": f"Excellent! {matched_milk} coming up."}

    def add_extra(self, extra: str) -> dict:
        """
        Add an extra/topping to the order.
        
        Logic:
        1. Normalize input (case-insensitive matching)
        2. Validate against VALID_EXTRAS
        3. Check if already added (avoid duplicates)
        4. Append to order.extras list
        5. Return feedback message
        
        Args:
            extra: The extra/topping to add
            
        Returns:
            Dictionary with success status and message
        """
        matched_extra = None
        for valid_extra in self.VALID_EXTRAS:
            if extra.lower() in valid_extra.lower():
                matched_extra = valid_extra
                break

        if not matched_extra:
            error_msg = f"Available extras: {', '.join(self.VALID_EXTRAS[:4])}..."
            logger.warning(f"Invalid extra requested: {extra}")
            return {"success": False, "message": error_msg}

        # Don't add duplicates
        if matched_extra in self.order.extras:
            return {
                "success": False,
                "message": f"{matched_extra} is already in your order.",
            }

        self.order.extras.append(matched_extra)
        logger.info(f"Extra added: {matched_extra}")
        return {
            "success": True,
            "message": f"Added {matched_extra} to your order.",
        }

    def remove_extra(self, extra: str) -> dict:
        """
        Remove an extra/topping from the order.
        
        Logic:
        1. Find matching extra (case-insensitive)
        2. Check if it's in the current order
        3. Remove it from order.extras list
        4. Return feedback message
        
        Args:
            extra: The extra to remove
            
        Returns:
            Dictionary with success status and message
        """
        matched_extra = None
        for valid_extra in self.VALID_EXTRAS:
            if extra.lower() in valid_extra.lower():
                matched_extra = valid_extra
                break

        if not matched_extra or matched_extra not in self.order.extras:
            return {
                "success": False,
                "message": f"That extra is not in your order.",
            }

        self.order.extras.remove(matched_extra)
        logger.info(f"Extra removed: {matched_extra}")
        return {
            "success": True,
            "message": f"Removed {matched_extra} from your order.",
        }

    def set_customer_name(self, name: str) -> dict:
        """
        Set the customer's name for the order.
        
        Logic:
        1. Validate name is not empty
        2. Clean up the name (strip whitespace, capitalize)
        3. Update order.name
        4. Return feedback message
        
        Args:
            name: The customer's name
            
        Returns:
            Dictionary with success status and message
        """
        if not name or len(name.strip()) == 0:
            return {
                "success": False,
                "message": "Please provide a valid name.",
            }

        cleaned_name = name.strip()
        self.order.name = cleaned_name
        logger.info(f"Customer name set to: {cleaned_name}")
        return {
            "success": True,
            "message": f"Got it, {cleaned_name}! Your order will be ready soon.",
        }

    def get_current_order(self) -> dict:
        """
        Get the current order state.
        
        Logic:
        1. Convert OrderState dataclass to dictionary
        2. Return all current field values
        3. Used by LLM to understand what it knows about the order
        
        Returns:
            Dictionary representation of current order
        """
        return asdict(self.order)

    def is_order_complete(self) -> bool:
        """
        Check if all required fields are filled.
        
        Logic:
        1. Check that all required fields have values (not None or empty)
        2. Returns True only if complete
        
        Returns:
            True if all fields are set, False otherwise
        """
        return all(
            [
                self.order.drinkType is not None,
                self.order.size is not None,
                self.order.milk is not None,
                self.order.extras is not None,  # Can be empty list
                self.order.name is not None,
            ]
        )

    def get_missing_fields(self) -> list[str]:
        """
        Get list of fields that still need to be filled.
        
        Logic:
        1. Check each field
        2. If None, add to missing list
        3. Return the list for LLM to know what to ask for next
        
        Returns:
            List of field names that are missing
        """
        missing = []
        if self.order.drinkType is None:
            missing.append("drink type")
        if self.order.size is None:
            missing.append("size")
        if self.order.milk is None:
            missing.append("milk option")
        if self.order.name is None:
            missing.append("name")
        return missing

    def save_order_to_json(self) -> str:
        """
        Save completed order to a JSON file.
        
        Logic:
        1. Check if order is complete (fail fast if not)
        2. Generate unique filename with timestamp and customer name
        3. Create order object with metadata (ID, timestamp, status)
        4. Write to JSON file in orders/ directory
        5. Return filename for confirmation
        
        Returns:
            Filename of saved order
            
        Raises:
            ValueError: If order is not complete
        """
        if not self.is_order_complete():
            missing = self.get_missing_fields()
            raise ValueError(f"Cannot save incomplete order. Missing: {missing}")

        # Generate unique order ID and filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        order_id = f"ORD-{timestamp}"
        filename = f"order_{timestamp}_{self.order.name.replace(' ', '_')}.json"
        filepath = self.orders_dir / filename

        # Prepare order data with metadata
        order_data = {
            "orderId": order_id,
            "timestamp": datetime.now().isoformat(),
            "customer": {"name": self.order.name},
            "order": asdict(self.order),
            "status": "completed",
        }

        # Write to JSON file
        with open(filepath, "w") as f:
            json.dump(order_data, f, indent=2)

        logger.info(f"Order saved to: {filepath}")
        return filename

    def reset_order(self):
        """
        Reset the order state for a new customer.
        
        Logic:
        1. Create a new empty OrderState
        2. This is called after an order is completed
        3. Prepares manager for next customer
        """
        self.order = OrderState()
        logger.info("Order reset for new customer")
