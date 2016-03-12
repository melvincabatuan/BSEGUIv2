package ph.edu.dlsu.fx.utils;

import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javafx.stage.StageStyle;

/**
 * Created by cobalt on 3/6/16.
 */
public class ConfirmationBox {

    static Stage stage;
    static boolean yesButtonClicked;

    public static boolean show(String message, String textYes, String textNo){
        yesButtonClicked = false;

        stage = new Stage();
        stage.initModality(Modality.APPLICATION_MODAL);
        stage.initStyle(StageStyle.UNDECORATED);
        stage.setMinWidth(250);

        Label label = new Label();
        label.setFont(Font.font("Times New Roman", FontWeight.SEMI_BOLD, 30));
        label.setTextFill(Color.BLACK);
        label.setText(message);

        Button yesButton = new Button();
        yesButton.setText(textYes);
        yesButton.setFont(Font.font("Times New Roman", FontWeight.SEMI_BOLD, 25));
        yesButton.setMaxWidth(Double.MAX_VALUE);
        yesButton.setOnAction(e -> onClickYes());

        Button noButton = new Button();
        noButton.setText(textNo);
        noButton.setFont(Font.font("Times New Roman", FontWeight.SEMI_BOLD, 25));
        noButton.setMaxWidth(Double.MAX_VALUE);
        noButton.setOnAction(e -> onClickNo());

        HBox paneBtn = new HBox(20);
        paneBtn.getChildren().addAll(yesButton, noButton);
        paneBtn.setPadding(new Insets(20));
        paneBtn.setAlignment(Pos.CENTER);

        VBox pane = new VBox(20);
        pane.getChildren().addAll(label, paneBtn);
        pane.setPadding(new Insets(20));
        pane.setAlignment(Pos.CENTER);

        Scene scene = new Scene(pane);
        scene.setFill(Color.BLACK);
        stage.setScene(scene);
        stage.showAndWait();
        return yesButtonClicked;
    }

    private static void onClickNo() {
        stage.close();
        yesButtonClicked = false;
    }

    private static void onClickYes() {
        stage.close();
        yesButtonClicked = true;
    }

}
