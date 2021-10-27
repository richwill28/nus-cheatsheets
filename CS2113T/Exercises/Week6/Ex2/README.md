## \[Key Exercise\] weekly roster

The class given below keeps track of how many people signup to attend an event on each day of the week. Add the missing methods so that it produces the output given.

💡 Use an `HashMap` to store the number of entries for each day.

```java
public class Main {
    private static HashMap<String, Integer> roster = new HashMap<>();

    //TODO: add your methods here

    public static void main(String[] args) {
        addToRoster("Monday"); // i.e., one person signed up for Monday
        addToRoster("Wednesday"); // i.e., one person signed up for Wednesday
        addToRoster("Wednesday"); // i.e., another person signed up for Wednesday
        addToRoster("Friday");
        addToRoster("Monday");
        printRoster();
    }
}
```

⬇️

```console
Monday => 2
Friday => 1
Wednesday => 2
```

<details>
  <summary>Hint</summary>

  Partial solution:

  ```java
  import java.util.HashMap;
  import java.util.Map;

  public class Main {
    private static HashMap<String, Integer> roster = new HashMap<>();

      private static void addToRoster(String day) {
          if (roster.containsKey(day)){
              Integer newValue = Integer.valueOf(roster.get(day).intValue() + 1);
              roster.put(day, newValue);
          } else {
              roster.put(day, Integer.valueOf(1));
          }
      }

      // ...
  }
  ```
</details>

Website: https://nus-cs2113-ay2122s1.github.io/website/schedule/week6/topics.html#key-exercise-weekly-roster
