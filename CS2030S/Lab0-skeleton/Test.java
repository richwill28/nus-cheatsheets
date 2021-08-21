class Test {

  public static void main(String[] args) {

    // Test Point
    expect("Point: new at (0, 0)",
        new Point(0, 0).toString(),
        "(0.0, 0.0)");

    // Test Circle
    expect("Circle: new at (0, 0) with radius 4)",
        new Circle(new Point(0, 0), 4).toString(),
        "{ center: (0.0, 0.0), radius: 4.0 }");

    expect("Circle: does {(0, 0), 4} contains (0, 0)?",
        new Circle(new Point(0, 0), 4)
        .contains(new Point(0, 0)),
        true);

    expect("Circle: does {(0, 0), 4} contains (4, 4)?",
        new Circle(new Point(0, 0), 4)
        .contains(new Point(4, 4)),
        false);

    // Test RandomPoint
    expect("RandomPoint: new with default seed",
        new RandomPoint(0, 1, 0, 1).toString(),
        "(0.7308781907032909, 0.41008081149220166)");

    RandomPoint.setSeed(10);
    expect("RandomPoint: new with seed 10",
        new RandomPoint(0, 1, 0, 1).toString(),
        "(0.7304302967434272, 0.2578027905957804)");

    expect("RandomPoint: next with the same seed",
        new RandomPoint(0, 1, 0, 1).toString(),
        "(0.059201965811244595, 0.24411725056425315)");

    RandomPoint.setSeed(10);
    expect("RandomPoint: reset seed to 10 and new again",
        new RandomPoint(0, 1, 0, 1).toString(),
        "(0.7304302967434272, 0.2578027905957804)");
  }

  public static final String ANSI_RESET = "\u001B[0m";
  public static final String ANSI_RED = "\u001B[31m";
  public static final String ANSI_GREEN = "\u001B[32m";

  public static void expect(String test, Object output, Object expect) {
    System.out.print(test);
    if (output.equals(expect)) {
      System.out.println(".. " + ANSI_GREEN + "ok" + ANSI_RESET);
    } else {
      System.out.println(".. " + ANSI_RED + "failed" + ANSI_RESET);
      System.out.println("  expected: " + expect);
      System.out.println("  got this: " + output);
    }
  }
}
