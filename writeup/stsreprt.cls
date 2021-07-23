\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{stsreprt}
\RequirePackage{kvoptions}
\DeclareBoolOption[false]{nobcorignoretitle}
\DeclareBoolOption[false]{grayscalebody}
\DeclareBoolOption[false]{grayscaletitle}
\DeclareBoolOption[false]{fancyheadings}
\DeclareStringOption[scrbook]{baseclass}
\DeclareStringOption[TUHH-STS]{titlestyle}
\DeclareStringOption[]{titleoptions}
\ProcessKeyvalOptions*
\LoadClassWithOptions{\stsreprt@baseclass}

\ifstsreprt@fancyheadings
\RequirePackage[automark,headsepline,footsepline]{scrlayer-scrpage}
\else
\RequirePackage[automark]{scrpage2}
\fi

\RequirePackage{graphicx}
\RequirePackageWithOptions{xcolor}
\RequirePackage{titlepage}
\ifx\stsreprt@titlestyle\@empty
\else
\TitlePageStyle[\stsreprt@titleoptions]{\stsreprt@titlestyle}
\fi

\pagestyle{scrheadings}
\ifstsreprt@fancyheadings
	\newcommand{\twosideheadings}{\rehead{\leftmark}\lohead{\rightmark}}
	\newcommand{\onesideheadings}{\ihead{\leftmark}\chead{}}

	\ifstsreprt@grayscalebody
		\ohead{\setlength{\unitlength}{0.5cm}\begin{picture}(1,0)\put(0,0){\includegraphics[width=0.5cm]{logo/logo_sts_letters_gray}}\end{picture}}
	\else
		\ohead{\setlength{\unitlength}{0.5cm}\begin{picture}(1,0)\put(0,0){\includegraphics[width=0.5cm]{logo/logo_sts_letters}}\end{picture}}
	\fi
	
	\makeatletter
	\if@twoside\twosideheadings \else \onesideheadings \fi
	\makeatother
	
	\ifx\chapterpagestyle\undefined\else\renewcommand*{\chapterpagestyle}{scrheadings}\fi
	\ifx\partpagestyle\undefined\else\renewcommand*{\partpagestyle}{scrheadings}\fi
	\ifx\indexpagestyle\undefined\else\renewcommand*{\indexpagestyle}{scrheadings}\fi
	\ifx\chapterheadstartvskip\undefined\else\renewcommand{\chapterheadstartvskip}{\vspace *{-\baselineskip }}\fi
\else
\fi


\endinput
