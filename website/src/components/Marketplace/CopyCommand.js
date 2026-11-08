import React, {useRef, useState} from 'react';
import s from './overview.module.css';

/**
 * A command box with Clio Coder's Copy control. Without clipboard access the
 * command is selected instead, so it can still be copied by hand.
 */
export default function CopyCommand({command, prompt = '$', label}) {
  const [state, setState] = useState('idle');
  const code = useRef(null);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setState('copied');
    } catch {
      const range = document.createRange();
      range.selectNodeContents(code.current);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      setState('selected');
    }
    setTimeout(() => setState('idle'), 1800);
  };
  return (
    <div className={s.command}>
      <code ref={code}>
        {command.split('\n').map((line, index) => (
          // eslint-disable-next-line react/no-array-index-key -- lines are static
          <span key={index}>
            {prompt && (
              <span className={s.commandPrompt} aria-hidden="true">
                {prompt}
              </span>
            )}
            {line}
            {'\n'}
          </span>
        ))}
      </code>
      <button
        type="button"
        className={s.copy}
        data-state={state}
        onClick={copy}
        aria-label={label || `Copy ${command.split('\n')[0]}`}
      >
        {state === 'copied' ? 'Copied' : state === 'selected' ? 'Selected' : 'Copy'}
      </button>
    </div>
  );
}
