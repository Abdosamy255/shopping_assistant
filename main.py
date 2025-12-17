# main.py

from nlp.preprocessing import preprocess_text
from nlp.attribute_extraction import extract_attributes
from search.search_engine import load_products, search_products

def main():
    df = load_products()

    while True:
        user_input = input("اكتب وصف المنتج (exit للخروج): ").strip()
        if user_input.lower() in ["exit", "خروج"]:
            break

        tokens, lang = preprocess_text(user_input)
        attrs = extract_attributes(tokens, lang)


        results = search_products(df, attrs, top_n=5)
        print("[3] Results:")
        if results.empty:
            print("❌ لا توجد نتائج مناسبة في قاعدة البيانات.")
        else:
            for i, row in results.iterrows():
                print(f"- {row['product_name']} | {row['brand']} | {row['price']} جنيه")

        print("\n" + "-"*40)

if __name__ == "__main__":
    main()
