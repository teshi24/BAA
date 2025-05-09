function RawInline(el)
  if el.format == "latex" then
    local text = el.text
    -- Suche nach \baaCriteria{...}
    local match = text:match("\\baaCriteria{(.*)}")
    if match then
      -- Erzeuge HTML-Span
      return pandoc.RawInline("html", '<span style="color:blue;">' .. match .. '</span>')
    end
  end
end



-- function RawInline(el)
  -- if el.format == "latex" then
    -- local color, text = el.text:match("\\textcolor%{(.-)%}%{(.-)%}")
    -- if color and text then
      -- return pandoc.Span(text, {["style"] = "color:" .. color})
    -- end
  -- end
-- end





-- local current_color = nil

-- -- Detects raw \color{...} changes
-- function RawInline(el)
  -- if el.format == "latex" then
    -- print("[RawInline] Raw LaTeX found:", el.text)

  
    -- -- Detect simple \color{colour}
    -- local color = el.text:match("\\color%{(.-)%}")
    -- if color then
      -- print("[ColorSwitch] Switching color to:", color)
      -- current_color = color
      -- return {} -- Remove the LaTeX command from output
    -- end

    -- -- Detect wrapped macros like \baaCriteria{Text}
    -- local macro, arg = el.text:match("\\(%a+)%{(.-)%}")
    -- if macro and arg then
	  -- print("[MacroMatch] Found macro:", macro, "with arg:", arg)
      -- if macro == "baaCriteria" then
        -- return pandoc.Span(arg, {style = "color:blue"})
      -- elseif macro == "todo" then
        -- return pandoc.Span("TODO: " .. arg, {style = "color:red"})
      -- end
	-- else
      -- print("[MacroMatch] No macro match for:", el.text)
    -- end
  -- end
-- end

-- -- Colours simple strings while a \color is active
-- function Str(el)
  -- if current_color then
    -- print("[Str] Coloring string:", el.text, "with color:", current_color)
    -- return pandoc.Span(el.text, {style = "color:" .. current_color})
  -- else
    -- return el
  -- end
-- end
