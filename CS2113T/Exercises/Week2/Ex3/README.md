## \[Key Exercise\] `getGradeCap` Method

Add the following method to the class given below.

- `public static double getGradeCap(String grade)`: Returns the CAP value of 
the given `grade`. The mapping from grades to CAP is given below.

| A+   | A    | A-   | B+   | B    | B-   | C    | Else |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 5.0  | 5.0  | 4.5  | 4.0  | 3.5  | 3.0  | 2.5  | 0.0  |

<mark>Do not change the code of the <code>main</code> method!</mark>

```java
public class Main {
    // ADD YOUR CODE HERE

    public static void main(String[] args) {
        System.out.println("A+: " + getGradeCap("A+"));
        System.out.println("B : " + getGradeCap("B"));
    }
}
```

⬇️

```console
A+: 5.0
B : 3.5
```

<details>
  <summary>Hint</summary>

  Partial solution:
  
  ```java
  public static double getGradeCap(String grade) {
      double cap = 0;
      switch (grade) {
      case "A+":
      case "A":
          cap = 5.0;
          break;
      case "A-":
          cap = 4.5;
          break;
      case "B+":
          cap = 4.0;
          break;
      case "B":
          cap = 3.5;
          break;
      case "B-":
          cap = 3.0;
          break;
      default:
      }
      return cap;
  }
  ```
</details>

Website: https://nus-cs2113-ay2122s1.github.io/website/schedule/week2/topics.html#key-exercise-getgradecap-method
