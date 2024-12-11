def generate_response(model, processor, prompt, image):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text = [text_prompt],
        images = [image],
        padding = True,
        return_tensors = "pt"
    )
    inputs = inputs.to("cuda")

    output_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids = [
        output_ids[len(input_ids) :] for input_ids, output_ids 
        in zip(inputs.input_ids, output_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_space=True
    )
    
    return output_text
