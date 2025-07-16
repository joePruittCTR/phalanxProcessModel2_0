import salabim as sb
import sys

print(f"Python executable: {sys.executable}")
print(f"Salabim version loaded: {sb.__version__}")
print(f"Salabim location: {sb.__file__}")

try:
    env = sb.Environment(trace=False, do_animate=True)
    print("\nsb.Environment with do_animate=True created successfully.")
except TypeError as e:
    print(f"\nError creating environment: {e}")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")

# If successful, try a minimal run to ensure it's fully functional
if 'env' in locals():
    class MyComponent(sb.Component):
        def process(self):
            print(f"Component running at time {self.env.now()}")
            yield self.hold(1)
            print(f"Component finished at time {self.env.now()}")

    MyComponent()
    try:
        env.run(till=2)
        print("Minimal simulation run completed successfully.")
    except Exception as e:
        print(f"Error during minimal simulation run: {e}")

