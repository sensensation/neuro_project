def train_model(model, X_train, y_train, X_val, y_val):
    """Обучает модель."""
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)
    return history