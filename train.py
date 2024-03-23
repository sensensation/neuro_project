# def train_model(model, X_train, y_train, X_val, y_val):
#     """Обучает модель."""
#     model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#     history = model.fit(
#         X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32
#     )
#     return history

import asyncio


async def train_model_async(model, X_train, y_train, epochs=10):
    loop = asyncio.get_running_loop()
    fit = loop.run_in_executor(
        None,
        lambda: model.fit(
            X_train, y_train, epochs=epochs, validation_split=0.1, verbose=0
        ),
    )
    history = await fit
    return history
