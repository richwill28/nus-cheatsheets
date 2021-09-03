## \[Key Exercise\] define a `Circle` class

Define a `Circle` class so that the code given below produces the given output. 
The nature of the class is as follows:

- Attributes(all `private`):
    - `int x`, `int y`: represents the location of the circle
    - `double radius`: the radius of the circle
- Constructors:
    - `Circle()`: initializes `x`, `y`, `radius` to 0
    - `Circle(int x, int y, double radius)`: initializes the attributes to the given values
- Methods:
    - `getArea()`: `int`  
    Returns the area of the circle as an int value (not double). Calculated as Pi * (radius)<sup>2</sup>  
    💡 You can convert a `double` to an `int` using `(int)` e.g., `x = (int)2.25` gives `x` the value `2`.  
    💡 You can use `Math.PI` to get the value of Pi  
    💡 You can use `Math.pow()` to raise a number to a specific power e.g., `Math.pow(3, 2)` calculates <code>3<sup>2</sup></code>

```java
public class Main {
    public static void main(String[] args) {
        Circle c = new Circle();

        System.out.println(c.getArea());
        c = new Circle(1, 2, 5);
        System.out.println(c.getArea());
    }
}
```

⬇️

```console
0
78
```

<details>
  <summary>Hint</summary>

  - Put the `Circle` class in a file called `Circle.java`

  Partial solution:

  ```java
  public class Circle {
      private int x;
      // ...

      public Circle() {
          this(0, 0, 0);
      }

      public Circle(int x, int y, double radius) {
          this.x = x;
          // ...
      }

      public int getArea() {
          double area = Math.PI * Math.pow(radius, 2);
          return (int)area;
      }
  }
  ```
</details>

Website: https://nus-cs2113-ay2122s1.github.io/website/schedule/week3/topics.html#key-exercise-define-a-circle-class
