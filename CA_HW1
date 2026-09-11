import argparse
import math


GRAVITY_VALUES = {
    "earth": 9.81,
    "moon": 1.62,
    "mars": 3.71,
    "jupiter": 24.79,
    "mercury": 3.70,
    "venus": 8.87,
}


def calculate_fall_time(height, gravity):
    return math.sqrt((2 * height) / gravity)


def main():

    parser = argparse.ArgumentParser(
        description="Calculate the time it takes a ball to reach the ground."
    )

    parser.add_argument(
        "-g",
        "--gravity",
        choices=GRAVITY_VALUES.keys(),
        default="earth",
        help="Choose a planet for gravity (default: earth)"
    )

    parser.add_argument(
        "--custom-gravity",
        type=float,
        help="Enter a custom gravity value in m/s^2"
    )

    args = parser.parse_args()

    # Ask the user for the height
    height = float(input("Enter the height of the ball in meters: "))

    if height < 0:
        print("Height cannot be negative.")
        return

    # Decide which gravity to use
    if args.custom_gravity is not None:
        gravity = args.custom_gravity
        gravity_name = "Custom"
    else:
        gravity = GRAVITY_VALUES[args.gravity]
        gravity_name = args.gravity.capitalize()

    # Calculate time
    time = calculate_fall_time(height, gravity)

    print()
    print(f"Height: {height:.2f} meters")
    print(f"Gravity: {gravity_name} ({gravity:.2f} m/s^2)")
    print(f"Time to reach the ground: {time:.2f} seconds")


if __name__ == "__main__":
    main()
