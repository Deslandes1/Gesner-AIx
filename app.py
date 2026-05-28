def generate_response(user_input, uploaded_image_bytes=None):
    try:
        if uploaded_image_bytes:
            return identify_image_with_grok(uploaded_image_bytes, user_input), False, False

        if not user_input or not user_input.strip():
            return "Tanpri ekri yon kestyon.", True, False

        q = user_input.strip()

        # 1. CORE ANSWERS (instant)
        core_answer = get_core_answer(q)
        if core_answer:
            return core_answer, False, False

        # 2. DIRECT RULES
        direct = direct_keyword_answer(q)
        if direct:
            return direct, False, False

        # 3. SMALL TALK FIX (prevents "stuck feeling")
        small = small_talk_response(q)
        if small:
            return small, False, False

        # 4. MATH / REASONING
        math_result = reason_about_question(q)
        if math_result:
            return math_result, False, False

        # 5. COGNITIVE MEMORY
        try:
            cog_match = find_cognitive_match(q)
            if cog_match:
                return apply_cognitive_format(q, cog_match), False, False
        except:
            pass

        # 6. FACT RETRIEVAL (FORCE RESULT)
        facts = retrieve_facts_hybrid(q, k=10)

        if facts and len(facts) > 0:
            # ALWAYS force response (fixes stuck empty box)
            answer = reason_answer(q, facts)
            if answer and answer.strip():
                return answer, False, False

        # 7. GROK FALLBACK (with safety retry logic)
        try:
            grok_answer = call_grok_api(q)
            if grok_answer and grok_answer.strip():
                return grok_answer, False, False
        except:
            pass

    except Exception:
        pass

    # 8. GUARANTEED NEVER-EMPTY FALLBACK (CRITICAL FIX)
    fallback_pool = [
        "Mwen pa fin konprann kestyon an, men eseye eksplike li yon lòt fason 😊",
        "Mwen la toujou, men mwen bezwen plis detay pou reponn sa 👍",
        "Sa pa klè pou mwen, ou ka reekri kestyon an?",
        "Mwen poko gen ase enfòmasyon pou sa, men mwen ap aprann 📚",
        "Tanpri eseye poze kestyon an yon lòt fason 😊"
    ]

    return random.choice(fallback_pool), True, False
