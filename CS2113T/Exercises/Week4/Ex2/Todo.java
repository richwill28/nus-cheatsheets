public class Todo extends Task {
    protected boolean isDone;

    public Todo(String description) {
        super(description);
        isDone = false;
    }

    public void setDone(boolean done) {
        isDone = done;
    }

    public boolean isDone() {
        return isDone;
    }

    @Override
    public String toString() {
        String status = null;
        if (isDone) {
            status = "Yes";
        } else {
            status = "No";
        }
        return super.toString() + System.lineSeparator() + "is done? " + status;
    }
}
