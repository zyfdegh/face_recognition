package org.tensorflow.lite.examples.detection;

public class SettingItem {
    private String name;
    private String description;
    private String hint;

    public SettingItem(String name, String description, String hint) {
        this.name = name;
        this.description = description;
        this.hint = hint;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public String getHint() {
        return hint;
    }

    public void setHint(String hint) {
        this.hint = hint;
    }

}
