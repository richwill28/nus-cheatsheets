public class Deadline extends Todo {
    protected String by;

    public String getBy() {
        return by;
    }

    public void setBy(String deadline) {
        by = deadline;
    }

    public Deadline(String description, String by) {
        super(description);
        this.by = by;
    }
}
