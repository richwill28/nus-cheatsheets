## \[Key Exercise\] Numbers list

Add the missing methods to the class given below so that it produces the output given.

💡 Use an `ArrayList` to store the numbers.

```java
public class Main {
    //TODO: add your methods here

    public static void main(String[] args) {
        System.out.println("Adding numbers to the list");
        addNumber(3);
        addNumber(8);
        addNumber(24);
        System.out.println("The total is: " + getTotal());
        System.out.println("8 in the list : " + isFound(8) );
        System.out.println("5 in the list : " + isFound(5) );
        removeNumber(8);
        System.out.println("The total is: " + getTotal());
    }
}
```

⬇️

```console
Adding numbers to the list
[3]
[3, 8]
[3, 8, 24]
The total is: 35
8 in the list : true
5 in the list : false
[3, 24]
The total is: 27
```

<details>
  <summary>Hint</summary>

  Partial solution:

  ```java
  import java.util.ArrayList;

  public class Main {
      private static ArrayList<Integer> numbers = new ArrayList<>();

      private static void addNumber(int i) {
          numbers.add(Integer.valueOf(i));
          System.out.println(numbers);
      }

      // ...
  }
  ```
</details>

Website: https://nus-cs2113-ay2122s1.github.io/website/schedule/week6/topics.html#key-exercise-numbers-list
